"""Estimate paired sample savings and recall differences for matched scan additions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nki_calibration import PRIMARY, analyze_condition, crossings


ROOT = Path(__file__).resolve().parents[1]
PEOPLE = {"nki": 25, "bnu1": 57}
COARSE = {0, 8, 16, 32, 64, 128, 256, 512, 2352}
METHODS = {"pool_plus_own": "personal", "pool_plus_other": "pooled", "pool_only": "scratch"}
KEY = ["subject", "candidate_draw", "other_subject", "repeat", "budget", "mix_rule", "method"]
SCORES = ["r2", "nmse"]


def read_scores(path):
    frame = pd.read_csv(path, dtype={"subject": str, "other_subject": str})
    if frame.duplicated(KEY).any() or not np.isfinite(frame[SCORES]).all().all():
        raise ValueError(f"Duplicate cells or nonfinite scores in {path}")
    if set(frame.source_visit) != {1} or set(frame.target_visit) != {2}:
        raise ValueError(f"Expected first to second visit transfer in {path}")
    return frame


def check_complete(frame, n):
    people = set(frame.subject)
    if len(people) != n or set(frame.method) != set(METHODS):
        raise ValueError("Unexpected methods or participants")
    expected_grid = COARSE | set(range(32, 65, 4))
    if set(frame["repeat"]) != set(range(6)) or not expected_grid <= set(frame.budget):
        raise ValueError("Missing calibration repetitions or budgets")
    if set(frame.mix_rule) != {"equal_donor", "fixed_025"} or set(frame.candidate_draw) != {0, 1, 2}:
        raise ValueError("Expected both prespecified allocations and three donor assignments")
    pairings = frame[["subject", "candidate_draw", "other_subject"]].drop_duplicates()
    if (pairings.subject == pairings.other_subject).any():
        raise ValueError("Own subject present in unrelated addition")
    if pairings.duplicated(["subject", "candidate_draw"]).any() or pairings.duplicated(["subject", "other_subject"]).any():
        raise ValueError("A sampled participant has inconsistent or repeated unrelated donors")
    for _, draw in pairings.groupby("candidate_draw"):
        if set(draw.subject) != people or set(draw.other_subject) != people:
            raise ValueError("Unbalanced unrelated-donor assignments")
    expected = len(pairings) * 6 * frame.budget.nunique() * frame.mix_rule.nunique() * len(METHODS)
    if len(frame) != expected:
        raise ValueError("Incomplete cells across participants, donors, methods or budgets")
    # The bank-only control has the same training rows under both allocations.
    keys = ["subject", "candidate_draw", "repeat", "budget"]
    bank = frame[frame.method == "pool_only"].set_index(keys).sort_index()
    np.testing.assert_allclose(bank.loc[bank.mix_rule == "equal_donor", SCORES],
                               bank.loc[bank.mix_rule == "fixed_025", SCORES], atol=1e-9, rtol=1e-9)


def merge_grids(coarse, fine_path):
    if not fine_path.exists():
        return coarse
    fine = read_scores(fine_path)
    coarse_budgets = set(coarse.budget)
    overlap = fine[fine.budget.isin(coarse_budgets)].merge(coarse, on=KEY, validate="one_to_one")
    expected_overlap = coarse.budget.isin(set(fine.budget)).sum()
    if len(overlap) != expected_overlap:
        raise ValueError("Refined and original grids disagree on repeated cells")
    for score in SCORES:
        np.testing.assert_allclose(overlap[f"{score}_x"], overlap[f"{score}_y"], atol=1e-9, rtol=1e-9)
    return pd.concat([coarse, fine[~fine.budget.isin(coarse_budgets)]], ignore_index=True)


def original_names(frame):
    lookup = {short: full for full, short in METHODS.items()}
    for column in ("method", "method_a", "method_b"):
        if column in frame:
            frame[column] = frame[column].map(lookup)
    return frame


def summarize_rule(scores, cohort, rule):
    # Give the three unrelated donor choices equal weight inside a target's repeat.
    # The sampling unit for both confidence intervals and crossings is a person.
    group = ["subject", "repeat", "budget", "method"]
    balanced = scores.groupby(group, as_index=False)[SCORES].mean()
    balanced["method"] = balanced.method.map(METHODS)
    stats = analyze_condition(balanced, {"cohort": cohort, "mix_rule": rule}, expected_n=PEOPLE[cohort])
    stats = {key: original_names(pd.DataFrame(rows)) for key, rows in stats.items()}

    primary = stats["threshold_comparisons"].query(
        "method_a == 'pool_plus_own' and method_b == 'pool_plus_other' and threshold == 0.5"
    ).iloc[0].to_dict()
    accuracy = stats["low_budget"].query(
        "method_a == 'pool_plus_own' and method_b == 'pool_plus_other'"
    ).iloc[0].to_dict()
    draws = []
    for donor_draw, group in scores.groupby("candidate_draw"):
        table = group.groupby(["method", "budget"]).r2.mean().unstack().sort_index(axis=1)
        pair = table.loc[["pool_plus_own", "pool_plus_other"]]
        own, other = crossings(pair.to_numpy(), table.columns.to_numpy(), .50)
        advantage = pair.loc[:, list(PRIMARY)].mean(axis=1)
        draws.append({"candidate_draw": int(donor_draw), "own_first_sustained": own,
                      "other_first_sustained": other, "examples_saved_by_own": other - own,
                      "own_minus_other_low_budget_mean_r2": advantage.iloc[0] - advantage.iloc[1]})

    summary = {"threshold": primary, "low_budget": accuracy, "donor_draw_sensitivity": draws}
    return stats, summary


def write_outputs(cohort, folder):
    coarse = read_scores(folder / "scores.csv")
    scores = merge_grids(coarse, folder / "refinement" / "scores.csv")
    check_complete(scores, PEOPLE[cohort])
    tables = {}
    summary = {"cohort": cohort, "participants": PEOPLE[cohort],
               "target_budgets": sorted(int(budget) for budget in scores.budget.unique()),
               "low_budgets": list(PRIMARY), "bootstrap_unit": "participant",
               "bootstrap_count": 10000, "bootstrap_seed": 20261120, "mix_rules": {}}
    for rule, by_rule in scores.groupby("mix_rule"):
        stats, rule_summary = summarize_rule(by_rule, cohort, rule)
        summary["mix_rules"][rule] = rule_summary
        for filename, table in (
            ("curves.csv", stats["curves"]),
            ("thresholds.csv", stats["threshold_comparisons"]),
            ("person_crossings.csv", stats["participant_thresholds"]),
            ("contrasts.csv", stats["budget_contrasts"]),
            ("low_budget_contrasts.csv", stats["low_budget"]),
        ):
            tables.setdefault(filename, []).append(table)
    for filename, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(folder / filename, index=False)
    safe_json = pd.Series(summary, dtype=object).to_json(double_precision=15)
    (folder / "summary.json").write_text(json.dumps(json.loads(safe_json), indent=2) + "\n", encoding="utf-8")
    for rule, result in summary["mix_rules"].items():
        t, a = result["threshold"], result["low_budget"]
        print(cohort, rule, "examples (own, other):", t["cohort_a_grid_budget"], t["cohort_b_grid_budget"],
              "| own fewer, equal, more:", t["fewer_for_a"], t["same_tested_budget"], t["more_for_a"],
              "| mean own - other R2 [paired 95% CI]:", a["mean_r2_difference"], a["ci_low"], a["ci_high"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=PEOPLE, nargs="+")
    args = parser.parse_args()
    for name in args.cohort:
        write_outputs(name, ROOT / "results" / f"{name}-augmentation")
