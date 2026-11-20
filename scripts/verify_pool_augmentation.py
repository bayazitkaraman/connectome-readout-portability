"""Audit matched first-scan substitutions on the two repeat-scan cohorts."""
import argparse
import hashlib
import json
from pathlib import Path

from verify_results import SingularRidge, fixtures, join, sampled, stream
from verify_nki_calibration import GRID, adjacency, audit_prediction
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "nki": ROOT / "data/nki/nki_reference.npz",
    "bnu1": ROOT / "data/replication_candidates/corr/bnu1_dkt_retest.npz",
}
OWN, OTHER, BANK = "pool_plus_own", "pool_plus_other", "pool_only"
RULES = ("equal_donor", "fixed_025")
INDEX = ["subject", "candidate_draw", "mix_rule", "repeat", "budget", "method"]


def source_data(cohort):
    source = DATA[cohort]
    with np.load(source, allow_pickle=False) as data:
        graphs = data["weights"]
        if cohort == "nki":
            assert data["halves"][0] == "odd"
            graphs = graphs[0]
        subjects = data["subjects"].astype(str).tolist()
    assert graphs.shape[:2] == (2, len(subjects))
    assert len(subjects) == len(set(subjects))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    signed = np.zeros(graphs.shape[-1])
    nodes = stream(10).choice(len(signed), 8, replace=False)
    signed[nodes[:4]], signed[nodes[4:]] = .5, -.5
    return subjects, graphs, signed, digest


def sustained(budgets, values):
    """Return the first budget after the last below-threshold point."""
    failed = np.flatnonzero(values < .50)
    i = 1 + failed[-1] if len(failed) else 0
    return float(budgets[i]) if i < len(budgets) else np.nan


def reconstruct_endpoint(scores, own, other):
    budgets = np.sort(scores.budget.unique())
    grouped = scores.groupby(["subject", "method", "budget"]).r2.mean()
    tables = {m: grouped.xs(m, level="method").unstack("budget").sort_index()
              for m in [own, other]}
    assert tables[own].index.equals(tables[other].index)
    assert np.array_equal(budgets, tables[own].columns)
    curves = {m: t.mean(axis=0).to_numpy() for m, t in tables.items()}
    starts = {m: sustained(budgets, c) for m, c in curves.items()}
    individuals = {m: np.array([sustained(budgets, row) for row in t.to_numpy()])
                   for m, t in tables.items()}
    gains = individuals[other] - individuals[own]
    n = len(gains)
    counts = np.random.default_rng(20261120).multinomial(n, np.ones(n) / n, size=10_000)
    boots = {m: np.array([sustained(budgets, row) for row in counts @ t.to_numpy() / n])
             for m, t in tables.items()}
    differences = boots[other] - boots[own]
    interval = np.sort(differences)[[249, 9749]].tolist() if np.isfinite(differences).all() else None
    matched = [8, 16, 32, 64, 128, 256]
    by_person = (tables[own][matched] - tables[other][matched]).mean(axis=1).to_numpy()
    accuracy_interval = np.quantile(counts @ by_person / n, [.025, .975])
    return {
        "cohort_threshold": starts,
        "steps_saved_by_personal": starts[other] - starts[own],
        "paired_interval": interval,
        "own_fewer_equal_more": [int(np.sum(gains > 0)), int(np.sum(gains == 0)), int(np.sum(gains < 0))],
        "people_both_attained": int(np.isfinite(gains).sum()),
        "positive_grid": budgets[budgets > 0].tolist(),
        "mean_own_minus_other_low_budget_r2": float(by_person.mean()),
        "low_budget_accuracy_interval": accuracy_interval.tolist(),
    }


def verify_threshold_file(out, endpoint, rule, own, other):
    thresholds = pd.read_csv(out / "thresholds.csv")
    row = thresholds[(thresholds.mix_rule == rule) & (thresholds.threshold == .50) & (thresholds.method_a == own)
                     & (thresholds.method_b == other)].iloc[0]
    np.testing.assert_equal(
        [row.cohort_a_grid_budget, row.cohort_b_grid_budget, row.cohort_grid_saving_for_a,
         row.fewer_for_a, row.same_tested_budget, row.more_for_a],
        [endpoint["cohort_threshold"][own], endpoint["cohort_threshold"][other],
         endpoint["steps_saved_by_personal"], *endpoint["own_fewer_equal_more"]])
    if endpoint["paired_interval"] is not None:
        np.testing.assert_equal([row.cohort_grid_saving_ci_low, row.cohort_grid_saving_ci_high],
                                endpoint["paired_interval"])
    accuracy = pd.read_csv(out / "low_budget_contrasts.csv")
    paired = accuracy[(accuracy.mix_rule == rule) & (accuracy.method_a == own) &
                      (accuracy.method_b == other)].iloc[0]
    np.testing.assert_allclose([paired.mean_r2_difference, paired.ci_low, paired.ci_high],
                               [endpoint["mean_own_minus_other_low_budget_r2"],
                                *endpoint["low_budget_accuracy_interval"]], atol=1e-12, rtol=1e-10)


def allocate(indices, budget, subjects):
    base, extra = divmod(budget, len(indices))
    return {subjects[j]: int(base + (pos < extra)) for pos, j in enumerate(indices)}


def verify_records(records, subjects):
    n = len(subjects)
    order = stream(172, n).permutation(n).tolist()
    rows = {}
    for record in records:
        person, draw = record["subject"], record["candidate_draw"]
        j = subjects.index(person)
        s = order[(order.index(j) + draw + 1) % n]
        other = subjects[s]
        assert record["other_subject"] == other and other != person
        bank = [i for i in range(n) if i not in (j, s)]
        assert set(record["common_bank_donors"]) == {subjects[i] for i in bank}
        shuffled = stream(173, n, j, s).permutation(bank)
        nfit = round(2 * len(bank) / 3)
        assert record["inner_fit"] == allocate(shuffled[:nfit], 2352, subjects)
        assert record["inner_validation"] == allocate(shuffled[nfit:], 1176, subjects)
        assert record["bank_only_rows"] == allocate(bank, 2352, subjects)
        for rule, candidate_rows in zip(RULES, (round(2352/(n-1)), 588)):
            assert record["added_rows_by_rule"][rule] == candidate_rows
            assert record["common_rows_by_rule"][rule] == allocate(bank, 2352-candidate_rows, subjects)
        key = person, draw
        assert key not in rows
        rows[key] = record
    assert len(rows) == n * 3
    # Every index is used once in each offset, with three distinct substitutions per person.
    assert set(rows) == {(s, d) for s in subjects for d in range(3)}
    for draw in range(3):
        assert sorted(rows[(s, draw)]["other_subject"] for s in subjects) == sorted(subjects)
    assert all(len({rows[(s, d)]["other_subject"] for d in range(3)}) == 3 for s in subjects)
    return rows


def measured_sources(graphs, signed):
    standard, validation, auxiliary = [], [], []
    for i, raw in enumerate(graphs[0]):
        matrix = adjacency(raw)
        standard.append(fixtures(matrix, signed, 10_000 + i, 0, 2352))
        validation.append(fixtures(matrix, signed, 10_000 + i, 1, 1176))
        auxiliary.append(fixtures(matrix, signed, 29000, 80, 2352))
    # The auxiliary source task inputs and past-input targets must match across people.
    assert all(np.array_equal(auxiliary[0][1], other[1]) for other in auxiliary)
    return standard, validation, auxiliary


def reconstruct_priors(sources, subjects, rec, rule):
    standard, validation, auxiliary = sources
    def design(allocation, data):
        return [sampled(data[subjects.index(person)], rows) for person, rows in allocation.items()]

    nested = SingularRidge(join(design(rec["inner_fit"], standard)))
    alpha = nested.tune(join(design(rec["inner_validation"], validation)), GRID)
    np.testing.assert_allclose(alpha, rec["pooled_alpha"], rtol=1e-12, atol=0)
    common = design(rec["common_rows_by_rule"][rule], standard)
    add = rec["added_rows_by_rule"][rule]
    starts = {BANK: SingularRidge(join(design(rec["bank_only_rows"], standard)))}
    for method, index in [(OWN, subjects.index(rec["subject"])), (OTHER, subjects.index(rec["other_subject"]))]:
        starts[method] = SingularRidge(join([*common, sampled(auxiliary[index], add)]))
    return {label: (lambda x, model=fit: model.predict(x, alpha)) for label, fit in starts.items()}


def reconstruct_scores(sources, subjects, graphs, signed, records, scores):
    rows = scores.set_index(INDEX).sort_index()
    largest = 0.
    checked = 0
    selected = []
    for j, draw in [(0, 0), (len(subjects)-1, 2)]:
        person = subjects[j]
        selected.append({"subject": person, "candidate_draw": draw})
        raw = adjacency(graphs[1, j])
        ident = 10_000 + len(subjects) + j
        xt, yt = join([fixtures(raw, signed, ident, 61, 512, f) for f in range(4)])
        for rule in RULES:
            priors = reconstruct_priors(sources, subjects, records[person, draw], rule)
            for repeat in (0, 5):
                # Exactly the same test sequences and four calibration prefixes for both additions.
                streams = [fixtures(raw, signed, ident, 60, 588, repeat*4+f) for f in range(4)]
                for budget in (0, 8, 48, 64, 128, 2352):
                    pieces = [(x[:budget//4], y[:budget//4]) for x,y in streams]
                    for method, prior in priors.items():
                        entry = rows.loc[(person, draw, rule, repeat, budget, method)]
                        _, error = audit_prediction(pieces, prior, xt, yt, entry)
                        largest = max(largest, float(error))
                        checked += 1
    return {"checked_target_donor_pairs": selected, "held_out_score_pairs_checked": checked,
            "max_abs_difference_r2_or_nmse": largest}


def load_results(folder, subjects, records):
    def read(relative):
        return pd.read_csv(folder / relative, dtype={"subject":str, "other_subject":str})
    coarse = read("scores.csv")
    fine = read("refinement/scores.csv")
    original = set(coarse.budget.unique())
    duplicate = fine[fine.budget.isin(original)].merge(coarse, on=INDEX, validate="one_to_one")
    assert len(duplicate) == fine.budget.isin(original).sum()
    for field in ("r2", "nmse"):
        np.testing.assert_allclose(duplicate[field+"_x"], duplicate[field+"_y"], rtol=1e-9, atol=1e-9)
    merged = pd.concat([coarse, fine[~fine.budget.isin(original)]])
    assert set(merged.subject) == set(subjects) and not merged.duplicated(INDEX).any()
    assert set(merged.method) == {OWN, OTHER, BANK}
    assert set(merged.mix_rule) == set(RULES)
    assert set(merged.repeat) == set(range(6))
    assert set(zip(merged.source_visit, merged.target_visit)) == {(1, 2)}
    assert np.isfinite(merged[["r2", "nmse"]]).all().all()
    for rule, grid in merged.groupby("mix_rule"):
        assert len(grid) == len(subjects) * 3 * 3 * 6 * grid.budget.nunique()
    for person, draw, other in merged[["subject","candidate_draw","other_subject"]].drop_duplicates().itertuples(index=False):
        assert records[person, draw]["other_subject"] == other
    # The contextual bank fit is independent of the candidate allocation.
    shared = merged[merged.method==BANK].pivot(index=[k for k in INDEX if k!="mix_rule"],
                                              columns="mix_rule", values="r2").dropna()
    np.testing.assert_allclose(shared[RULES[0]], shared[RULES[1]], atol=1e-10, rtol=1e-10)
    return merged


def run(cohort):
    subjects, graphs, signed, digest = source_data(cohort)
    folder = ROOT / "results" / f"{cohort}-augmentation"
    metadata = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    assert metadata["input_sha256"] == digest and metadata["subjects"] == subjects
    assert metadata["same_candidate_input"] == {"stream_id":29000, "role":80, "repeat":0}
    assert metadata["half"] == ("odd" if cohort == "nki" else "all")
    assert [metadata[k] for k in ("source_rows","source_validation_rows","calibration_repeats",
                                  "test_streams","test_steps_per_stream")] == [2352,1176,6,4,512]
    records = verify_records(metadata["records"], subjects)
    scores = load_results(folder, subjects, records)
    sources = measured_sources(graphs, signed)
    replicated = reconstruct_scores(sources, subjects, graphs, signed, records, scores)
    endpoints = {}
    for rule, grid in scores.groupby("mix_rule"):
        endpoint = reconstruct_endpoint(grid, OWN, OTHER)
        verify_threshold_file(folder, endpoint, rule, OWN, OTHER)
        endpoints[rule] = endpoint
    output = {"input_sha256":digest, "cohort":cohort, "donor_pools_verified":len(records),
              "endpoints":endpoints, **replicated}
    (folder / "independent_check.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n",
                                                  encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cohort", choices=DATA)
    run(cli.parse_args().cohort)
