"""Paired uncertainty for own and average-stranger linear recall retention."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/portability-path"
FROZEN = ROOT / "results/density-controls/frozen"
BOOTSTRAP_SEED = 20261120
N_DRAWS = 10000


def contrasts(means):
    """Means can be one cohort or a stack of participant-bootstrap means."""
    own_source, own_target, stranger_source, stranger_target = means.T
    if np.any(own_source <= 0) or np.any(stranger_source <= 0):
        raise ValueError("A retention share has a nonpositive source denominator.")
    own = 100 * own_target / own_source
    stranger = 100 * stranger_target / stranger_source
    return {
        "own_share_pct": own,
        "stranger_share_pct": stranger,
        "own_minus_stranger_pp": own - stranger,
        "own_minus_stranger_r2": own_target - stranger_target,
    }


def main():
    people = pd.read_csv(OUT / "visit_people.csv.gz", dtype={"target_id": str})
    previous = pd.read_csv(FROZEN / "summary.csv").query("source_input == 'original_distinct_inputs'").set_index("cohort")
    original_people = pd.read_csv(FROZEN / "single_donor_participant_means.csv", dtype={"target_id": str})
    original_people = original_people.query("source_input == 'original_distinct_inputs'")
    records = []
    for cohort, alpha in (("nki", 10.), ("bnu1", 1.)):
        selected = people.loc[people.cohort.eq(cohort) & people.task.eq("linear") & people.alpha.eq(alpha)]
        wide = selected.pivot(index="target_id", columns="rung", values=["source_r2", "target_r2"]).sort_index()
        values = wide[[("source_r2", "between_visit"), ("target_r2", "between_visit"),
                       ("source_r2", "stranger"), ("target_r2", "stranger")]].to_numpy()
        n = len(wide)
        first = previous.loc[cohort]
        assert n == first.participants and np.isfinite(values).all()
        original = original_people.loc[original_people.cohort.eq(cohort)].set_index("target_id").sort_index()
        np.testing.assert_array_equal(wide.index, original.index)
        np.testing.assert_allclose(values[:, [1, 3]], original[["own_r2", "other_r2"]], rtol=0, atol=1e-12)
        # Reproduce the original frozen single-donor participant draws. Each target
        # contributes both arms, with its observed donor bank and test inputs fixed.
        weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.ones(n) / n, size=N_DRAWS)
        means = values.mean(axis=0)
        estimates = contrasts(means)
        boot = contrasts(weights @ values / n)
        row = dict(cohort=cohort, alpha=alpha, n_people=n, donors_per_target=n-1,
                   own_source_r2=means[0], own_target_r2=means[1], stranger_source_r2=means[2],
                   stranger_target_r2=means[3], bootstrap_seed=BOOTSTRAP_SEED, draws=N_DRAWS)
        for name, value in estimates.items():
            row[name] = value
            row[f"{name}_lo"], row[f"{name}_hi"] = np.quantile(boot[name], [.025, .975])
        np.testing.assert_allclose(
            [row["own_minus_stranger_r2"], row["own_minus_stranger_r2_lo"], row["own_minus_stranger_r2_hi"]],
            first[["mean_gain", "gain_ci_low", "gain_ci_high"]].to_numpy(dtype=float), atol=1e-12, rtol=0)
        records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "shares_summary.csv", index=False)
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
