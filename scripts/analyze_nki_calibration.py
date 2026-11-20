"""Paired estimates of accuracy and calibration budgets on a repeat scan."""
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "results" / "nki-calibration"
CONDITION = ["half", "source_visit", "target_visit"]
METHODS = ["personal", "pooled", "scratch"]
PAIRS = [("personal", "pooled"), ("personal", "scratch"), ("pooled", "scratch")]
PRIMARY = (8, 16, 32, 64, 128, 256)
THRESHOLDS = (.50, .80, .90)
MARGIN = .01


def bootstrap(matrix, counts):
    """Percentile intervals holding calibration draws and fitted donor banks fixed."""
    sampled = (counts / counts.sum(axis=1, keepdims=True)) @ matrix
    return np.quantile(sampled, (.025, .975), axis=0)


def crossings(curves, budgets, level):
    """First grid point where mean recall remains above level through the largest budget."""
    sustained = np.logical_and.accumulate((curves >= level)[:, ::-1], axis=1)[:, ::-1]
    return np.where(sustained.any(axis=1), budgets[sustained.argmax(axis=1)], np.nan)


def practical_conclusion(lo, hi):
    if lo > MARGIN:
        return "favors_first_beyond_margin"
    if hi < -MARGIN:
        return "favors_second_beyond_margin"
    if lo >= -MARGIN and hi <= MARGIN:
        return "interval_within_margin"
    return "uncertain_at_margin"


def analyze_condition(scores, condition, expected_n=25):
    cell = ["subject", "repeat", "budget", "method"]
    if scores.duplicated(cell).any() or not np.isfinite(scores[["r2", "nmse"]].to_numpy()).all():
        raise ValueError(f"Repeated cells or nonfinite results in {condition}")
    budgets = np.sort(scores.budget.unique())
    people = sorted(scores.subject.unique())
    n = len(people)
    repeats = scores.repeat.unique()
    if (n != expected_n or len(repeats) != 6 or set(scores.method) != set(METHODS)
            or budgets[0] != 0 or budgets[-1] != 2352 or not set(PRIMARY) <= set(budgets)
            or len(scores) != n*len(repeats)*len(budgets)*len(METHODS)):
        raise ValueError(f"Incomplete calibration grid in {condition}")
    means = scores.groupby(["subject", "budget", "method"])[["r2", "nmse"]].mean()

    def matrix(method, metric):
        return means.xs(method, level="method")[metric].unstack("budget").loc[people, budgets].to_numpy()

    r2 = {method: matrix(method, "r2") for method in METHODS}
    nmse = {method: matrix(method, "nmse") for method in METHODS}
    counts = np.random.default_rng(20261120).multinomial(n, np.ones(n)/n, size=10_000)
    curves, contrasts, primary, by_person = [], [], [], []
    thresholds, individual, compared = [], [], []

    for method in METHODS:
        r, e = r2[method], nmse[method]
        r_int, e_int = bootstrap(r, counts), bootstrap(e, counts)
        for ix, budget in enumerate(budgets):
            curves.append(dict(**condition, method=method, budget=budget, people=n,
                               mean_r2=r[:, ix].mean(), r2_ci_low=r_int[0, ix], r2_ci_high=r_int[1, ix],
                               mean_nmse=e[:, ix].mean(), nmse_ci_low=e_int[0, ix], nmse_ci_high=e_int[1, ix]))

    mask = np.isin(budgets, PRIMARY)
    for a, b in PAIRS:
        labels = dict(method_a=a, method_b=b)
        dr, de = r2[a] - r2[b], nmse[a] - nmse[b]
        ir, ie = bootstrap(dr, counts), bootstrap(de, counts)
        for ix, budget in enumerate(budgets):
            contrasts.append(dict(**condition, **labels, budget=budget, r2_difference=dr[:, ix].mean(),
                                  r2_ci_low=ir[0, ix], r2_ci_high=ir[1, ix], r2_wins=int((dr[:, ix]>0).sum()),
                                  nmse_difference=de[:, ix].mean(), nmse_ci_low=ie[0, ix], nmse_ci_high=ie[1, ix]))
        delta = dr[:, mask].mean(axis=1)
        lo, hi = bootstrap(delta, counts)
        primary.append(dict(**condition, **labels, mean_r2_difference=delta.mean(), ci_low=lo, ci_high=hi,
                            first_wins=int((delta>0).sum()), first_wins_by_margin=int((delta>MARGIN).sum()),
                            second_wins_by_margin=int((delta < -MARGIN).sum()),
                            margin=MARGIN, margin_conclusion=practical_conclusion(lo, hi)))
        for person, x in zip(people, delta):
            by_person.append(dict(**condition, subject=person, **labels, low_budget_r2_difference=x))

    resampled_curves = {m: (counts / n) @ r2[m] for m in METHODS}
    for level in THRESHOLDS:
        personal_times, group_times, bootstrap_times = {}, {}, {}
        for method in METHODS:
            curve = r2[method]
            times = crossings(curve, budgets, level)
            cohort_time = crossings(curve.mean(axis=0, keepdims=True), budgets, level)[0]
            personal_times[method], group_times[method] = times, cohort_time
            bootstrap_times[method] = crossings(resampled_curves[method], budgets, level)
            thresholds.append(dict(**condition, method=method, threshold=level, cohort_first_sustained=cohort_time,
                                   n_attaining=int(np.isfinite(times).sum()), n_right_censored=int(np.isnan(times).sum())))
            for person, x in zip(people, times):
                individual.append(dict(**condition, subject=person, method=method, threshold=level,
                                       first_sustained_budget=x, attained=bool(np.isfinite(x))))
        for a, b in PAIRS:
            x, y = personal_times[a], personal_times[b]
            both = np.isfinite(x) & np.isfinite(y)
            savings = y[both] - x[both]
            grid_difference = bootstrap_times[b] - bootstrap_times[a]
            finite = np.isfinite(grid_difference)
            # Give interval bounds only if the endpoint was reached by both methods
            # in every replicate. The grid is discrete, so keep observed quantiles.
            lo, hi = (np.quantile(grid_difference, (.025, .975), method="inverted_cdf")
                      if finite.all() else (np.nan, np.nan))
            compared.append(dict(**condition, method_a=a, method_b=b, threshold=level,
                                 cohort_a_grid_budget=group_times[a], cohort_b_grid_budget=group_times[b],
                                 cohort_grid_saving_for_a=group_times[b]-group_times[a],
                                 both_attained=int(both.sum()), fewer_for_a=int((savings>0).sum()),
                                 same_tested_budget=int((savings==0).sum()), more_for_a=int((savings<0).sum()),
                                 only_a_attained=int((np.isfinite(x)&np.isnan(y)).sum()),
                                 only_b_attained=int((np.isnan(x)&np.isfinite(y)).sum()),
                                 neither_attained=int((np.isnan(x)&np.isnan(y)).sum()),
                                 bootstrap_both_attained_fraction=finite.mean(),
                                 bootstrap_same_grid_fraction=float(np.mean(grid_difference == 0)),
                                 cohort_grid_saving_ci_low=lo, cohort_grid_saving_ci_high=hi,
                                 median_grid_saving_among_both=float(np.median(savings)) if len(savings) else np.nan))

    # Independent grouping of the raw grid: compare method grand means after selecting
    # the six positive budgets, bypassing the participant pivot and contrast code.
    simple = scores[scores.budget.isin(PRIMARY)].groupby("method").r2.mean()
    np.testing.assert_allclose([row["mean_r2_difference"] for row in primary],
                               [simple[a]-simple[b] for a, b in PAIRS], atol=1e-12, rtol=1e-11)
    return dict(curves=curves, budget_contrasts=contrasts, low_budget=primary,
                low_budget_participants=by_person, thresholds=thresholds,
                participant_thresholds=individual, threshold_comparisons=compared)


def main():
    scores = pd.read_csv(OUT/"scores.csv", dtype={"subject": str})
    conditions = list(scores.groupby(CONDITION, sort=False))
    if {key for key, _ in conditions} != {(h, s, 3-s) for h in ("odd", "even") for s in (1, 2)}:
        raise ValueError("Expected two diffusion halves and both transfer directions")
    output = {}
    for keys, frame in conditions:
        for name, rows in analyze_condition(frame, dict(zip(CONDITION, keys))).items():
            output.setdefault(name, []).extend(rows)
    for name, rows in output.items():
        pd.DataFrame(rows).to_csv(OUT/f"{name}.csv", index=False)
    print("Verified all low-budget point estimates independently against raw method averages.")


if __name__ == "__main__":
    main()
