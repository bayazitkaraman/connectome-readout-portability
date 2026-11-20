"""Fit paired pooled readouts with the target's or a matched donor's first scan."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import nki_fewshot as calibration
import nki_transfer as transfer
import replicate_calibration as replication
import numpy as np
import pandas as pd


pilot = transfer.pilot
ROOT = transfer.ROOT
DATA = {"nki": transfer.DEFAULT_DATA,
        "bnu1": ROOT / "data/replication_candidates/corr/bnu1_dkt_retest.npz"}
RULES = ("equal_donor", "fixed_025")
METHODS = ("pool_plus_own", "pool_plus_other", "pool_only")
METHOD_NAMES = dict(zip(calibration.METHODS, METHODS))
COARSE = calibration.BUDGETS
FINE = tuple(range(32, 65, 4))


def read_graphs(cohort):
    if cohort == "nki":
        graphs, halves, _, subjects, _ = transfer.load(DATA[cohort])
        return graphs[list(halves).index("odd")], subjects, "odd"
    graphs, subjects, _ = replication.load_pairs(DATA[cohort])
    return graphs, subjects, "all"


def split_rows(indices, rows):
    """Allocate any remainder in the given order, preserving all input indices."""
    quotient, remainder = divmod(rows, len(indices))
    return [(int(i), quotient + int(k < remainder)) for k, i in enumerate(indices)]


def readable(allocation, subjects):
    return {str(subjects[i]): int(rows) for i, rows in allocation}


def partners(n):
    order = pilot.rng(172, n).permutation(n)
    candidates = np.empty((n, 3), dtype=int)
    for draw in range(3):
        candidates[order, draw] = np.roll(order, -(draw + 1))
    if any(len(set(row)) != 3 or i in row for i, row in enumerate(candidates)):
        raise ValueError("A target must be matched to three distinct other people.")
    np.testing.assert_array_equal(np.bincount(candidates.ravel()), np.full(n, 3))
    return candidates


def source_streams(graphs):
    bank, validation, candidate = [], [], []
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    for i, graph in enumerate(graphs[0]):
        net = pilot.reservoir(graph, transfer.GAIN)
        bank.append(pilot.simulate(net, mapping, 10000+i, 0, pilot.FIT_N))
        validation.append(pilot.simulate(net, mapping, 10000+i, 1, pilot.VALID_N))
        candidate.append(pilot.simulate(net, mapping, 29000, 80, pilot.FIT_N))
        np.testing.assert_array_equal(candidate[0][1], candidate[-1][1])
    return bank, validation, candidate


def make_jobs(graphs, subjects, half):
    bank, validation, candidate = source_streams(graphs)
    n = len(subjects)
    matches = partners(n)

    @lru_cache(None)
    def bank_moment(i, rows):
        return pilot.Moment.from_data(pilot.sampled(bank[i], rows))

    @lru_cache(None)
    def added_moment(i, rows):
        return pilot.Moment.from_data(pilot.sampled(candidate[i], rows))

    def bank_stats(allocation):
        return pilot.merge([bank_moment(i, rows) for i, rows in allocation])

    jobs, metadata = [], []
    for i, subject in enumerate(subjects):
        for draw, other in enumerate(matches[i]):
            other = int(other)
            common = np.array([k for k in range(n) if k != i and k != other])
            shuffled = pilot.rng(173,n,i,other).permutation(common)
            cut = int(round(len(common) * 2 / 3))
            inner = split_rows(shuffled[:cut], pilot.FIT_N)
            valid = split_rows(shuffled[cut:], pilot.VALID_N)
            pooled_validation = pilot.combine([pilot.sampled(validation[k], rows) for k, rows in valid])
            inner_model = pilot.RidgeFamily(bank_stats(inner))
            alpha = inner_model.tune(pooled_validation, transfer.PENALTIES)
            only_rows = split_rows(common, pilot.FIT_N)
            bank_only = pilot.RidgeFamily(bank_stats(only_rows))
            common_log, added_log = {}, {}
            for rule in RULES:
                rows_added = int(round(pilot.FIT_N/(n-1))) if rule == "equal_donor" else pilot.FIT_N//4
                rows_common = split_rows(common, pilot.FIT_N - rows_added)
                base = bank_stats(rows_common)
                assert base.count + rows_added == pilot.FIT_N
                models = [pilot.RidgeFamily(pilot.merge([base, added_moment(k, rows_added)]))
                          for k in (i, other)]
                models.append(bank_only)
                slopes = [model.weights(alpha) for model in models]
                intercepts = [model.my - model.mx @ w for model, w in zip(models, slopes)]
                job = (half,1,2,str(subject),i,n,graphs[1,i],np.stack(slopes,axis=1),
                       np.stack(intercepts))
                jobs.append((draw,str(subjects[other]),rule,job))
                common_log[rule] = readable(rows_common, subjects)
                added_log[rule] = rows_added
            metadata.append(dict(subject=str(subject), other_subject=str(subjects[other]),
                                 candidate_draw=draw, common_bank_donors=subjects[common].tolist(),
                                 inner_fit=readable(inner,subjects), inner_validation=readable(valid,subjects),
                                 pooled_alpha=alpha, bank_only_rows=readable(only_rows,subjects),
                                 common_rows_by_rule=common_log, added_rows_by_rule=added_log))
    return jobs, metadata


def score_one(task):
    draw, other, rule, job, budgets = task
    scores, delays = calibration.simulate_person((*job, budgets))
    scores = pd.DataFrame(scores, columns=replication.SCORE_COLUMNS)
    delays = pd.DataFrame(delays, columns=replication.LAG_KEYS+["r2","nmse"])
    delays = delays.groupby(replication.LAG_KEYS, sort=False, as_index=False)[["r2","nmse"]].mean()
    for frame in (scores, delays):
        frame["method"] = frame.method.map(METHOD_NAMES)
        frame["candidate_draw"] = draw
        frame["other_subject"] = other
        frame["mix_rule"] = rule
    return scores, delays


def score(pool, jobs, budgets):
    frames, delay_frames = [], []
    for rows, lags in pool.map(score_one, [(*job,budgets) for job in jobs]):
        frames.append(rows)
        delay_frames.append(lags)
    scores, delays = pd.concat(frames,ignore_index=True), pd.concat(delay_frames,ignore_index=True)
    expected = len(jobs)*len(METHODS)*len(budgets)
    if len(scores) != expected*calibration.REPEATS or len(delays) != expected*len(pilot.DELAYS):
        raise ValueError("Missing or extra scores.")
    keys = ["subject","candidate_draw","mix_rule","method","budget"]
    if scores.duplicated(keys+["repeat"]).any() or delays.duplicated(keys+["delay"]).any():
        raise ValueError("Duplicate score cells.")
    return scores, delays


def extra_budgets(scores):
    primary = scores[scores.mix_rule == "equal_donor"]
    coarse = primary[primary.budget.isin(COARSE)].groupby(["method","budget"]).r2.mean().unstack()
    coarse_budgets = coarse.columns.to_numpy()
    times = replication.crossings(coarse.loc[list(METHODS[:2])].to_numpy(),coarse_budgets,.50)
    if not np.isfinite(times).all() or max(times)>128:
        return ()
    tested = set(primary.budget)
    wanted = set()
    for upper in times.astype(int):
        if upper > 0:
            lower = int(coarse_budgets[coarse_budgets < upper][-1])
            wanted.update(range(lower,upper+1,4))
    return tuple(sorted(wanted-tested))


def run(cohort, workers):
    graphs, subjects, half = read_graphs(cohort)
    print(f"{cohort}: preparing pooled source models for {len(subjects)} targets.",flush=True)
    jobs, records = make_jobs(graphs, subjects, half)
    base = tuple(sorted(set(COARSE)|set(FINE)))
    print(f"{cohort}: scoring {len(jobs)} target/donor/allocation combinations on {len(base)} budgets.",flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scores, delays = score(pool, jobs, base)
        extra = extra_budgets(scores)
        if extra:
            print(f"{cohort}: scoring added four-step points {extra}.",flush=True)
            more, more_delays = score(pool, jobs, extra)
            scores = pd.concat([scores,more],ignore_index=True)
            delays = pd.concat([delays,more_delays],ignore_index=True)
    out = ROOT / "results" / f"{cohort}-augmentation"
    for folder, budgets in ((out,COARSE),(out/"refinement",(*FINE,*extra))):
        replication.save_scores(folder,scores[scores.budget.isin(budgets)],delays[delays.budget.isin(budgets)])
    meta = dict(cohort=cohort, half=half, input_sha256=hashlib.sha256(DATA[cohort].read_bytes()).hexdigest(),
                subjects=subjects.tolist(), methods=METHODS, mix_rules=RULES, source_visit=1, target_visit=2,
                source_rows=pilot.FIT_N, source_validation_rows=pilot.VALID_N,
                same_candidate_input=dict(stream_id=29000, role=80, repeat=0),
                original_coarse_budgets=COARSE, refinement_budgets=(*FINE,*extra),
                candidate_draws=3, calibration_repeats=calibration.REPEATS, test_streams=pilot.TEST_STREAMS,
                test_steps_per_stream=pilot.TEST_N, seed=pilot.SEED, reservoir_gain=transfer.GAIN,
                ridge_grid=transfer.PENALTIES.tolist(), records=records)
    (out/"run_info.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"{cohort}: saved {len(scores)} unique scores for {len(subjects)} targets.",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",nargs="+",choices=tuple(DATA),default=list(DATA))
    parser.add_argument("--workers",type=int,default=7)
    args = parser.parse_args()
    for cohort in args.cohort:
        run(cohort, args.workers)
