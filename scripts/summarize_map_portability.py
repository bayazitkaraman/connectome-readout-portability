"""Summarize frozen recall across the ten evaluated native input maps, using participants as units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "headline-robustness"
MAPS = tuple(range(10, 20))
BOOTSTRAP_SEED = 20261120
N_DRAWS = 10_000
MEASURES = ["source_r2", "noise_r2", "repeat_r2", "excess_r2", "shift_pct",
            "noise_loss", "projected_loss", "residual_cross_term"]


def scores_from_means(means):
    """Accept one set of cohort means or one per bootstrap sample."""
    scores = dict(zip(MEASURES, means.T))
    scores["retention_pct"] = 100 * scores["repeat_r2"] / scores["source_r2"]
    scores["noise_retention_pct"] = 100 * scores["noise_r2"] / scores["source_r2"]
    spatial_excess = scores["projected_loss"] - scores["noise_loss"]
    scores["spatial_share_pct"] = 100 * spatial_excess / scores["excess_r2"]
    return scores


def map_summary(cohort, setting, frame, keys, draws):
    # Average maps within participants first. The complete repeated-measures
    # design then gives equal weight to each person and to each included map.
    by_person = frame[frame.map_key.isin(keys)].groupby("target_id", sort=True)[MEASURES].mean()
    values = by_person.to_numpy()
    point = scores_from_means(values.mean(axis=0))
    samples = scores_from_means(values[draws].mean(axis=1))
    row = dict(cohort=cohort, setting=setting, n_people=len(by_person), n_maps=len(keys),
               n_positive_excess=int((by_person.excess_r2 > 0).sum()))
    for name, estimate in point.items():
        row[name] = estimate
        row[name + "_low"], row[name + "_high"] = np.quantile(samples[name], [.025, .975])
    return row


def reverse_visit(cohort, directory):
    """Verify reverse-score differences from per-person predictions and recover the CLI interval."""
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    visits = list(map(str, run["visits"]))
    settings = run["settings"]
    assert len(visits) == 2 and settings["seeds"] == [20261119]
    assert settings["map_key"] == 10 and settings["gain"] == .9 and settings["task"] == "recall"
    read_ids = {"source_visit": str, "target_visit": str}
    people = pd.read_csv(directory / "state_pairs.csv", dtype={**read_ids, "subject": str})
    people = people[(people.source_visit == visits[1]) & (people.target_visit == visits[0])]
    ordered = people.set_index("subject").loc[run["subjects"]]
    differences = (ordered.isotropic_r2 - ordered.repeat_r2).to_numpy()
    np.testing.assert_allclose(differences, ordered.noise_minus_repeat, rtol=0, atol=1e-12)
    n = len(differences)
    # The CLI seeds its bootstrap by original signal seed and source and target visit positions.
    rng = np.random.default_rng(np.random.SeedSequence([settings["seeds"][0], 247, 1, 0]))
    draws = rng.integers(n, size=(N_DRAWS, n))
    lo, hi = np.quantile(differences[draws].mean(axis=1), [.025, .975])
    saved = pd.read_csv(directory / "state_summary.csv", dtype=read_ids)
    saved = saved[(saved.source_visit == visits[1]) & (saved.target_visit == visits[0])].iloc[0]
    np.testing.assert_allclose([differences.mean(), lo, hi],
                               [saved.noise_minus_repeat, saved.noise_minus_repeat_lo, saved.noise_minus_repeat_hi],
                               atol=1e-12, rtol=0)
    return dict(cohort=cohort, source_visit=visits[1], target_visit=visits[0], n=n,
                penalty_counts=ordered.penalty.value_counts().sort_index().to_dict(),
                excess_r2=float(differences.mean()), low=float(lo), high=float(hi))


def main():
    data = pd.read_csv(OUT / "people.csv", dtype={"target_id": str})
    selected = data[data.rule.eq("retuned")]
    summaries = []
    checks = dict(bootstrap_seed=BOOTSTRAP_SEED, draws=N_DRAWS, input_maps=list(MAPS),
                  unit="Participant; map-averaged summaries condition on the tested signed input maps.",
                  spatial_fraction="(mean projected loss - mean isotropic loss) / mean excess R2",
                  retention="mean measured repeat R2 / mean source R2",
                  cohorts={})
    choices = pd.read_csv(RESULTS / "calibration-budget-curve/penalty_selection.csv",
                          dtype={"target_id": str, "correction_penalty": str}, keep_default_na=False)
    for cohort, directory in [("nki", "cli-state-nki"), ("bnu1", "cli-state-beijing")]:
        cohort_rows = selected[selected.cohort.eq(cohort)]
        native = cohort_rows[cohort_rows.setting.isin([f"map_{key}" for key in MAPS])]
        sizes = native.groupby(["target_id","map_key"]).size().unstack()
        assert sizes.columns.tolist() == list(MAPS) and sizes.eq(1).all().all()
        assert native.representation.eq("native").all() and native.radius.eq(.9).all()
        n = len(sizes)
        assert n == {"nki":25, "bnu1":57}[cohort]
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(N_DRAWS, n))
        for label, keys in [("mean_ten_maps", MAPS), ("mean_nine_new_maps", MAPS[1:])]:
            summaries.append(map_summary(cohort, label, native, keys, draws))
        map_rows = [map_summary(cohort, f"map_{k}", native, [k], draws) for k in MAPS]
        summaries.extend(map_rows)
        by_map = pd.DataFrame(map_rows).set_index("setting")
        ranks = by_map[["excess_r2","repeat_r2","shift_pct","retention_pct"]].rank(ascending=False)
        variants = cohort_rows[~cohort_rows.setting.isin(by_map.index)].groupby("setting").excess_r2.mean()
        assert len(variants) == 5
        budget_counts = choices[choices.cohort.eq(cohort)].groupby("budget").correction_penalty.value_counts()
        correction = {int(b):budget_counts.loc[b].to_dict() for b in budget_counts.index.levels[0]}
        assert all(sum(counts.values()) == n for counts in correction.values())
        checks["cohorts"][cohort] = dict(
            n_people=n, original_map_rank_1_is_highest=ranks.loc["map_10"].astype(int).to_dict(),
            spatial_share_pct_min=float(by_map.spatial_share_pct.min()),
            spatial_share_pct_max=float(by_map.spatial_share_pct.max()),
            arithmetic_mean_map_spatial_shares_pct=float(by_map.spatial_share_pct.mean()),
            n_positive_native_map_means=int((by_map.excess_r2 > 0).sum()),
            n_positive_graph_variant_means=int((variants > 0).sum()),
            reverse_original_map=reverse_visit(cohort, RESULTS / directory),
            correction_penalty_counts_by_budget=correction)
    frame = pd.DataFrame(summaries)
    # Individual-map and nine-new-map means and intervals are independently archived.
    old = pd.read_csv(OUT / "summary.csv").query("rule == 'retuned'")
    shared = frame[frame.setting.ne("mean_ten_maps")].merge(old, on=["cohort","setting"],
                                                            validate="one_to_one", suffixes=("", "_archived"))
    assert len(shared) == 22
    for measure in MEASURES:
        for suffix in ("", "_low", "_high"):
            name = measure + suffix
            np.testing.assert_allclose(shared[name], shared[name + "_archived"], rtol=0, atol=1e-10)
    frame.to_csv(OUT / "map_portability.csv", index=False)
    (OUT / "reporting_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
