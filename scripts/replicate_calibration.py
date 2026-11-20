"""Apply the locked NKI calibration experiment to an independent two-visit cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import nki_fewshot as calibration
import nki_transfer as transfer
import numpy as np
import pandas as pd
from analyze_nki_calibration import crossings


pilot = transfer.pilot
IDENTIFIERS = ["half", "source_visit", "target_visit", "subject"]
SCORE_COLUMNS = IDENTIFIERS + ["repeat", "budget", "method", "r2", "nmse", "alpha"]
LAG_KEYS = IDENTIFIERS + ["budget", "method", "delay"]


def load_pairs(path):
    with np.load(path, allow_pickle=False) as data:
        subjects = data["subjects"].astype(str)
        graphs = data["weights"].astype(float, copy=True)
        labels = data["sessions"].astype(str) if "sessions" in data else np.array(["1", "2"])
    if subjects.ndim != 1 or len(subjects) < 3 or len(set(subjects)) != len(subjects):
        raise ValueError("Expected at least three unique participant identifiers.")
    if graphs.ndim != 4 or graphs.shape[:2] != (2, len(subjects)):
        raise ValueError("Expected weights ordered as two visits, participants and two region axes.")
    if graphs.shape[-1] != graphs.shape[-2] or graphs.shape[-1] <= 8:
        raise ValueError("The common parcellation must have more than eight regions.")
    if labels.shape != (2,) or labels[0] == labels[1]:
        raise ValueError("Expected two distinct visit labels.")
    if not np.isfinite(graphs).all() or (graphs < 0).any() or not np.array_equal(graphs, graphs.swapaxes(-1,-2)):
        raise ValueError("Expected finite, nonnegative and symmetric structural connectivity matrices.")
    idx = np.arange(graphs.shape[-1])
    graphs[..., idx, idx] = 0
    if np.any(graphs.sum(axis=(-1,-2)) <= 0):
        raise ValueError("Detected an empty interregional connectome.")
    return graphs, subjects, labels


@dataclass
class SourceFit:
    training: tuple
    validation: tuple
    model: pilot.RidgeFamily


def fit_visits(graphs, mapping):
    visits = []
    for visit, matrices in enumerate(graphs):
        scans = []
        for i, matrix in enumerate(matrices):
            key = 10000 + visit * len(matrices) + i
            net = pilot.reservoir(matrix, transfer.GAIN)
            training = pilot.simulate(net, mapping, key, 0, pilot.FIT_N)
            validation = pilot.simulate(net, mapping, key, 1, pilot.VALID_N)
            model = pilot.RidgeFamily.from_design(training)
            if not np.isfinite(model.scale) or model.scale <= 0:
                raise ValueError(f"Input placement produced zero response for visit {visit+1}, participant {i}.")
            scans.append(SourceFit(training, validation, model))
        visits.append(scans)
    return visits


def starting_models(visits, source_visit, graphs, subjects):
    """Construct each prior with the target excluded from its tuning participants."""
    target_visit = 1 - source_visit
    sources, targets = visits[source_visit], visits[target_visit]
    count = len(subjects)
    validation_errors = np.empty((count, len(transfer.PENALTIES)))
    for i, (source, target) in enumerate(zip(sources, targets)):
        for k, alpha in enumerate(transfer.PENALTIES):
            predicted = source.model.predict(target.validation[0], alpha)
            validation_errors[i,k] = transfer.errors(predicted, target.validation)[0].mean()
    jobs, prior_log = [], []
    for i, subject in enumerate(subjects):
        donor_mask = np.arange(count) != i
        alpha = transfer.PENALTIES[np.argmin(validation_errors[donor_mask].mean(axis=0))]
        personal = sources[i].model
        pooled, pooled_alpha, donor_log = transfer.shared_pool(sources, i, subjects)
        personal_weights, pooled_weights = personal.weights(alpha), pooled.weights(pooled_alpha)
        weights = np.stack([personal_weights, pooled_weights, np.zeros_like(personal_weights)], axis=1)
        intercepts = np.stack([personal.my - personal.mx @ personal_weights,
                               pooled.my - pooled.mx @ pooled_weights, np.zeros_like(personal.my)])
        jobs.append(("all", source_visit+1, target_visit+1, str(subject), i, count,
                     graphs[target_visit,i], weights, intercepts))
        prior_log.append(dict(subject=str(subject), source_visit=source_visit+1, target_visit=target_visit+1,
                              personal_alpha=float(alpha), pooled_alpha=float(pooled_alpha),
                              pooled_donors=donor_log))
    return jobs, prior_log


def score_jobs(pool, jobs, budgets):
    score_parts, lag_parts = [], []
    for rows, lags in pool.map(calibration.simulate_person, [(*job, budgets) for job in jobs]):
        score_parts.append(pd.DataFrame(rows, columns=SCORE_COLUMNS))
        lag_frame = pd.DataFrame(lags, columns=LAG_KEYS+["r2","nmse"])
        lag_parts.append(lag_frame.groupby(LAG_KEYS, as_index=False)[["r2","nmse"]].mean())
    scores = pd.concat(score_parts, ignore_index=True)
    delays = pd.concat(lag_parts, ignore_index=True)
    expected = len(jobs)*len(budgets)*len(calibration.METHODS)
    if len(scores)!=expected*calibration.REPEATS or len(delays)!=expected*len(pilot.DELAYS):
        raise RuntimeError("Incomplete scores for the evaluation grid.")
    if scores.duplicated(IDENTIFIERS+["budget","repeat","method"]).any() or delays.duplicated(LAG_KEYS).any():
        raise RuntimeError("Duplicate evaluation cell.")
    return scores, delays


def refinement_grid(scores):
    curves = scores.groupby(["method","budget"]).r2.mean().unstack("budget")
    coarse_budgets = curves.columns.to_numpy()
    required = crossings(curves.loc[["personal","pooled"]].to_numpy(), coarse_budgets, .50)
    if not (np.isfinite(required).all() and ((required > 0) & (required <= 128)).all()):
        return ()
    grid = set()
    for upper in required.astype(int):
        lower = coarse_budgets[coarse_budgets < upper][-1]
        grid.update(range(int(lower), int(upper)+1, 4))
    return tuple(sorted(grid))


def save_scores(out, scores, lags):
    out.mkdir(exist_ok=True, parents=True)
    scores.to_csv(out/"scores.csv", index=False)
    lags.to_csv(out/"delay_scores.csv", index=False)


def run(data, out, workers):
    calibration.check()
    graphs, subjects, labels = load_pairs(data)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    print(f"Preparing baselines for {len(subjects)} people, {graphs.shape[-1]} regions.", flush=True)
    visits = fit_visits(graphs, mapping)
    jobs = []
    prior_log = []
    for source_visit in (0,1):
        direction_jobs, log = starting_models(visits, source_visit, graphs, subjects)
        jobs += direction_jobs
        prior_log += log
    del visits

    with ProcessPoolExecutor(max_workers=workers) as pool:
        print("Scoring both visit directions.", flush=True)
        scores, lags = score_jobs(pool, jobs, calibration.BUDGETS)
        save_scores(out, scores, lags)
        primary = scores[scores.source_visit==1]
        fine = refinement_grid(primary)
        if fine:
            print(f"Scoring primary threshold refinement at {fine}.", flush=True)
            extra, extra_lags = score_jobs(pool, jobs[:len(subjects)], fine)
            match = extra.merge(primary, on=IDENTIFIERS+["budget","repeat","method"], validate="one_to_one")
            for metric in ["r2", "nmse"]:
                np.testing.assert_allclose(match[metric+"_x"], match[metric+"_y"], rtol=1e-9, atol=1e-9)
            save_scores(out/"refinement", extra, extra_lags)
    map_weights = mapping[0].ravel()
    meta = {
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "subjects": subjects.tolist(),
        "sessions": labels.tolist(), "regions": graphs.shape[-1], "n": len(subjects),
        "methods": calibration.METHODS, "budgets": calibration.BUDGETS, "refined_forward_budgets": fine,
        "directions": [[1,2],[2,1]], "repeats": calibration.REPEATS, "folds": calibration.FOLDS,
        "primary_endpoint": "first tested budget sustaining mean held-out R2 >= 0.50",
        "signed_input_weights_one_based": {str(i+1):float(map_weights[i]) for i in np.flatnonzero(map_weights)},
        "reservoir_gain":transfer.GAIN, "person_specific_source_rows": pilot.FIT_N,
        "pooled_source_rows": pilot.FIT_N, "pooled_tuning_rows":pilot.VALID_N,
        "personal_tuning_rows_per_donor":pilot.VALID_N, "n_calibration_sequences":calibration.FOLDS,
        "n_test_sequences":pilot.TEST_STREAMS, "test_sequence_length":pilot.TEST_N, "warmup":pilot.WARMUP,
        "calibration_rng_role":60, "test_rng_role":61, "personal_prior_selection":"other people's paired visits only",
        "relative_ridge_grid":transfer.PENALTIES.tolist(), "rng_seed":pilot.SEED, "all_fit_priors":prior_log,
    }
    (out/"run_info.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"Saved {len(scores)} coarse scores to {out}.",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path, required=True)
    parser.add_argument("--out",type=Path, required=True)
    parser.add_argument("--workers",type=int, default=6)
    args = parser.parse_args()
    run(args.data, args.out, args.workers)
