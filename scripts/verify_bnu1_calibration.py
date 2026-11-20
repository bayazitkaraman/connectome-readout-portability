"""Reconstruct BNU1 predictions independently and audit the forward savings endpoint."""
import hashlib
import json
from pathlib import Path

from verify_nki_calibration import GRID, adjacency, audit_prediction, investigate_sources
from verify_results import SingularRidge, fixtures, join, sampled, stream
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/replication_candidates/corr/bnu1_dkt_retest.npz"
OUT = ROOT / "results/bnu1-calibration"
METHODS = ("personal", "pooled", "scratch")
KEY = ["subject", "repeat", "budget", "method"]


def donor_groups(j, subjects):
    others = [i for i in range(len(subjects)) if i != j]
    shuffled = stream(70, j).permutation(others)
    split = round(len(others) * 2 / 3)
    groups = {}
    for label, indices, length in [
        ("final_fit", others, 2352),
        ("inner_fit", shuffled[:split], 2352),
        ("inner_validation", shuffled[split:], 1176),
    ]:
        rows, remainder = divmod(length, len(indices))
        groups[label] = {subjects[i]: rows + int(k < remainder) for k, i in enumerate(indices)}
    return groups


def direct_priors(sources, j, subjects, record):
    train, valid, fitted, losses = sources
    eligible = np.arange(len(subjects)) != j
    own_alpha = GRID[np.argmin(losses[eligible].mean(axis=0))]
    np.testing.assert_allclose(own_alpha, record["personal_alpha"], atol=0, rtol=1e-12)
    groups = donor_groups(j, subjects)

    def design(name, streams):
        return join([sampled(streams[subjects.index(person)], n) for person, n in groups[name].items()])

    nested = SingularRidge(design("inner_fit", train))
    group_alpha = nested.tune(design("inner_validation", valid), GRID)
    np.testing.assert_allclose(group_alpha, record["pooled_alpha"], atol=0, rtol=1e-12)
    group_fit = SingularRidge(design("final_fit", train))
    # Centered direct prediction reconstructs both slope and unpenalized intercept.
    return {
        "personal": lambda x: fitted[j].predict(x, own_alpha),
        "pooled": lambda x: group_fit.predict(x, group_alpha),
        "scratch": lambda x: np.zeros((len(x), 10)),
    }


def first_sustained(budgets, values):
    last_below = np.flatnonzero(values < .50)
    index = 0 if not len(last_below) else int(last_below[-1] + 1)
    return float(budgets[index]) if index < len(budgets) else np.nan


def check_savings(frame):
    budgets = np.sort(frame.budget.unique())
    averages = frame.groupby(["subject", "method", "budget"]).r2.mean()
    tables = {m: averages.xs(m, level="method").unstack("budget").sort_index() for m in METHODS}
    curves = {m: table.mean(axis=0).to_numpy() for m, table in tables.items()}
    cutoffs = {m: first_sustained(budgets, y) for m, y in curves.items()}
    people = {m: np.array([first_sustained(budgets, row) for row in tables[m].to_numpy()]) for m in METHODS}
    gains = people["pooled"] - people["personal"]
    n = len(gains)
    freq = np.random.default_rng(20261120).multinomial(n, np.full(n, 1/n), size=10000)
    boot = {m: np.array([first_sustained(budgets, row) for row in freq @ tables[m].to_numpy() / n])
            for m in ("personal", "pooled")}
    savings = boot["pooled"] - boot["personal"]
    assert np.isfinite(savings).all()
    # Use order statistics directly to audit the discrete bootstrap quantiles.
    limits = np.sort(savings)[[249, 9749]].tolist()
    observed = cutoffs["pooled"] - cutoffs["personal"]
    official = pd.read_csv(OUT / "refinement/threshold_comparisons.csv")
    summary = official.query("threshold==.50 and method_a=='personal' and method_b=='pooled'").iloc[0]
    counts = [int((gains > 0).sum()), int((gains == 0).sum()), int((gains < 0).sum())]
    np.testing.assert_equal(
        [cutoffs["personal"], cutoffs["pooled"], observed, *limits, *counts],
        [summary.cohort_a_grid_budget, summary.cohort_b_grid_budget, summary.cohort_grid_saving_for_a,
         summary.cohort_grid_saving_ci_low, summary.cohort_grid_saving_ci_high, summary.fewer_for_a,
         summary.same_tested_budget, summary.more_for_a])
    return {"budgets": budgets.tolist(), "sustained_at_0.50": cutoffs, "pooled_minus_personal": observed,
            "paired_bootstrap_95pct": limits, "people_fewer_equal_more": counts,
            "people_both_attaining": int(np.isfinite(gains).sum())}


def main():
    meta = json.loads((OUT / "run_info.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == meta["input_sha256"]
    with np.load(DATA, allow_pickle=False) as data:
        subjects = data["subjects"].astype(str).tolist()
        assert subjects == meta["subjects"] and data["sessions"].tolist() == ["1", "2"]
        weights = data["weights"]
    assert weights.shape == (2, len(subjects), 83, 83)
    signed = np.zeros(weights.shape[-1])
    nodes = stream(10).choice(len(signed), 8, replace=False)
    signed[nodes[:4]], signed[nodes[4:]] = .5, -.5
    assert {str(i+1):signed[i] for i in nodes} == meta["signed_input_weights_one_based"]
    records = {(r["source_visit"], r["subject"]):r for r in meta["all_fit_priors"]}
    assert len(records) == 2 * len(subjects)
    for (visit, person), record in records.items():
        assert record["target_visit"] == 3 - visit
        assert donor_groups(subjects.index(person), subjects) == record["pooled_donors"]

    def read(name):
        return pd.read_csv(OUT / name, dtype={"subject": str})

    scores = read("scores.csv")
    fine = read("refinement/scores.csv")
    primary = scores[scores.source_visit == 1]
    merged = pd.concat([primary, fine[~fine.budget.isin(primary.budget)]])
    assert not merged.duplicated(KEY).any()
    assert len(merged) == len(subjects)*6*len(METHODS)*merged.budget.nunique()
    assert set(merged.method) == set(METHODS) and set(merged.subject) == set(subjects)
    forward = check_savings(merged)

    rows = merged.set_index(KEY).sort_index()
    sources = investigate_sources(weights, subjects, signed)
    indices = (0, len(subjects)//2, len(subjects)-1)
    budgets = (0, 8, 60, 64, 128, 2352)
    count, largest = 0, 0.
    for j in indices:
        person = subjects[j]
        priors = direct_priors(sources, j, subjects, records[1, person])
        a = adjacency(weights[1, j])
        ident = 10000 + len(subjects) + j
        xt, yt = join([fixtures(a, signed, ident, 61, 512, f) for f in range(4)])
        for repeat in (0, 5):
            cal = [fixtures(a, signed, ident, 60, 588, 4*repeat+f) for f in range(4)]
            for n in budgets:
                parts = [(x[:n//4], y[:n//4]) for x,y in cal]
                for method in METHODS:
                    _, error = audit_prediction(parts, priors[method], xt, yt, rows.loc[(person, repeat, n, method)])
                    largest = max(largest, float(error))
                    count += 1
    report = {"input_sha256": meta["input_sha256"], "verified_donor_splits": len(records),
              "direct_svd_subjects": [subjects[j] for j in indices], "calibration_draws": [0,5],
              "reconstructed_budgets": list(budgets), "reconstructed_methods": list(METHODS),
              "score_pairs_checked": count, "max_absolute_score_discrepancy": largest,
              "forward": forward}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (OUT / "independent_check.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
