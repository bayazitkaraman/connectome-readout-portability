"""Test whether a more similar first-scan connectome predicts a larger eight-example recall gain."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "anatomy-advantage"
INPUTS = {
    "nki": PROJECT / "data" / "nki" / "nki_reference.npz",
    "bnu1": PROJECT / "data" / "replication_candidates" / "corr" / "bnu1_dkt_retest.npz",
}
PAIR = ["subject", "candidate_draw", "other_subject"]
RULES = ["equal_donor", "fixed_025"]
METHODS = ["pool_plus_own", "pool_plus_other"]
MEASURES = ["own_similarity", "other_similarity", "anatomy_advantage",
            "own_r2", "other_r2", "accuracy_advantage"]
BOOT_SEED = 20261119
BOOT_DRAWS = 10_000
BUDGET = 8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wiring_similarities(source, cohort, manifest):
    with np.load(source, allow_pickle=False) as archive:
        subjects = list(archive["subjects"].astype(str))
        weights = archive["weights"]
        if cohort == "nki":
            halves = list(archive["halves"].astype(str))
            weights = weights[halves.index("odd")]
    n = len(subjects)
    assert subjects == manifest["subjects"] and n == len(set(subjects))
    assert weights.shape[:2] == (2, n) and weights.shape[-1] == weights.shape[-2]
    assert (manifest["source_visit"], manifest["target_visit"]) == (1, 2)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_array_equal(weights, weights.swapaxes(-1, -2))
    u, v = np.triu_indices(weights.shape[-1], k=1)
    edges = weights[:, :, u, v].astype(np.float64)
    assert edges.shape == (2, n, len(u)) and (edges.std(axis=2) > 0).all()
    centered = edges - edges.mean(axis=2, keepdims=True)
    scaled = centered / np.linalg.norm(centered, axis=2, keepdims=True)
    # Row = target, column = first-scan candidate.
    cross = scaled[1] @ scaled[0].T
    np.testing.assert_allclose(cross[0, -1], stats.pearsonr(edges[1, 0], edges[0, -1]).statistic,
                               atol=1e-12)
    return pd.DataFrame(cross, index=subjects, columns=subjects), len(u)


def pair_data(cohort, folder, manifest, similarities):
    scores = pd.read_csv(folder / "scores.csv", dtype={"subject": str, "other_subject": str})
    assert set(scores.half) == {manifest["half"]} and set(scores.source_visit) == {1} and set(scores.target_visit) == {2}
    frame = scores.loc[(scores.budget == BUDGET) & scores.method.isin(METHODS)]
    group = PAIR + ["mix_rule", "method"]
    assignments = pd.DataFrame(manifest["records"])[PAIR]
    assert frame[group + ["repeat"]].duplicated().sum() == 0
    assert set(frame.mix_rule) == set(RULES) and set(frame["repeat"]) == set(range(6))
    assert frame.groupby(group).size().eq(6).all()
    assert set(assignments.itertuples(index=False, name=None)) == set(frame[PAIR].itertuples(index=False, name=None))
    assert set(assignments.subject) == set(similarities.index)
    assert assignments.groupby("subject").other_subject.nunique().eq(3).all()
    assert (assignments.subject != assignments.other_subject).all()
    assert len(frame) == len(similarities) * 3 * len(RULES) * len(METHODS) * 6
    accuracy = frame.groupby(group).r2.mean().unstack("method").rename(
        columns={"pool_plus_own": "own_r2", "pool_plus_other": "other_r2"}).reset_index()
    accuracy.columns.name = None
    accuracy["accuracy_advantage"] = accuracy.own_r2 - accuracy.other_r2
    assignments["own_similarity"] = [similarities.loc[s, s] for s in assignments.subject]
    assignments["other_similarity"] = [similarities.loc[s, d] for s, d in
                                       zip(assignments.subject, assignments.other_subject)]
    assignments["anatomy_advantage"] = assignments.own_similarity - assignments.other_similarity
    paired = assignments.merge(accuracy, on=PAIR, validate="one_to_many")
    assert np.isfinite(paired[MEASURES]).all().all()
    paired.insert(0, "cohort", cohort)
    return paired.sort_values(["cohort", "mix_rule", "subject", "candidate_draw"]).reset_index(drop=True)


def correlation_samples(numerator, xss, yss):
    denom = np.sqrt(xss * yss)
    return np.divide(numerator, denom, where=denom > 0,
                     out=np.full_like(np.asarray(denom, dtype=float), np.nan))


def rank_correlations(x, y):
    xr, yr = stats.rankdata(x, axis=-1), stats.rankdata(y, axis=-1)
    xr -= xr.mean(axis=-1, keepdims=True)
    yr -= yr.mean(axis=-1, keepdims=True)
    return correlation_samples((xr * yr).sum(axis=-1), (xr * xr).sum(axis=-1), (yr * yr).sum(axis=-1))


def result(cohort, rule, analysis, statistic, value, samples, n):
    good = np.isfinite(samples)
    # Report undefined draws explicitly. No interval if any replicate is undefined.
    lo, hi = np.quantile(samples, (.025, .975)) if good.all() else (np.nan, np.nan)
    return {"cohort": cohort, "mix_rule": rule, "analysis": analysis, "statistic": statistic,
            "estimate": float(value), "ci_low": lo, "ci_high": hi, "participants": n,
            "bootstrap_draws": BOOT_DRAWS, "valid_bootstrap_draws": int(good.sum())}


def associations(cohort, pairs, people):
    n = people.subject.nunique()
    draws = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_DRAWS, n))
    rows = []
    for rule in RULES:
        sample = people.loc[people.mix_rule == rule].sort_values("subject")
        x, y = sample.anatomy_advantage.to_numpy(), sample.accuracy_advantage.to_numpy()
        population = rank_correlations(x, y)
        direct = stats.spearmanr(x, y).statistic
        np.testing.assert_allclose(population, direct, atol=1e-12, equal_nan=True)
        draws_r = rank_correlations(x[draws], y[draws])
        rows.append(result(cohort, rule, "between_participants", "spearman", population, draws_r, n))

    rule = "fixed_025"
    sample = pairs.loc[pairs.mix_rule == rule]
    x = sample.pivot(index="subject", columns="candidate_draw", values="anatomy_advantage").sort_index().to_numpy()
    y = sample.pivot(index="subject", columns="candidate_draw", values="accuracy_advantage").sort_index().to_numpy()
    assert x.shape == y.shape == (n, 3)
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    xy, xx, yy = (xc * yc).sum(axis=1), (xc * xc).sum(axis=1), (yc * yc).sum(axis=1)
    full = correlation_samples(xy.sum(), xx.sum(), yy.sum())
    boot = correlation_samples(xy[draws].sum(axis=1), xx[draws].sum(axis=1), yy[draws].sum(axis=1))
    rows.append(result(cohort, rule, "within_participant", "within_target_centered_pearson", full, boot, n))
    return rows


def run():
    pair_frames, participant_frames, estimates, descriptions = [], [], [], []
    provenance = {}
    for cohort, source in INPUTS.items():
        folder = PROJECT / "results" / f"{cohort}-augmentation"
        manifest = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        assert sha256(source) == manifest["input_sha256"]
        assert manifest["half"] == ("odd" if cohort == "nki" else "all")
        correlations, n_edges = wiring_similarities(source, cohort, manifest)
        pairs = pair_data(cohort, folder, manifest, correlations)
        people = pairs.groupby(["cohort", "mix_rule", "subject"], as_index=False)[MEASURES].mean()
        estimates.extend(associations(cohort, pairs, people))
        for rule in RULES:
            p = people.loc[people.mix_rule == rule]
            q = pairs.loc[pairs.mix_rule == rule]
            descriptions.append({"cohort": cohort, "mix_rule": rule, "participants": len(p), "pairs": len(q),
                                 **{f"mean_{name}": p[name].mean() for name in MEASURES},
                                 "median_anatomy_advantage": p.anatomy_advantage.median(),
                                 "sd_anatomy_advantage": p.anatomy_advantage.std(),
                                 "sd_accuracy_advantage": p.accuracy_advantage.std(),
                                 "participants_own_more_similar": int((p.anatomy_advantage > 0).sum()),
                                 "pairs_own_more_similar": int((q.anatomy_advantage > 0).sum())})
        pair_frames.append(pairs)
        participant_frames.append(people)
        provenance[cohort] = {"unique_edges_per_scan": n_edges, "half": manifest["half"],
                              "source_graph_sha256": manifest["input_sha256"],
                              "scores_sha256": sha256(folder / "scores.csv"),
                              "augmentation_manifest_sha256": sha256(folder / "run_info.json")}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(pair_frames, ignore_index=True).to_csv(OUT / "pairs.csv", index=False)
    pd.concat(participant_frames, ignore_index=True).to_csv(OUT / "participants.csv", index=False)
    statistics = pd.DataFrame(estimates)
    statistics.to_csv(OUT / "statistics.csv", index=False)
    descriptives = pd.DataFrame(descriptions)
    descriptives.to_csv(OUT / "descriptives.csv", index=False)
    meta = {"classification": "exploratory follow-up to the pooled-augmentation result", "target_labels": BUDGET,
            "calibration_repetitions": 6, "matched_other_scans": 3, "comparison": "own first minus other first",
            "anatomy": "Pearson correlation of raw, upper-triangular, off-diagonal edge weights including zeros",
            "primary": "Spearman correlation across target people, equal_donor",
            "sensitivity": "same across-target statistic with a 25 percent source share",
            "secondary": "within-target centered Pearson with a 25 percent source share across three donors per target",
            "bootstrap_draws": BOOT_DRAWS, "bootstrap_seed": BOOT_SEED,
            "bootstrap_unit": "target person, conditional on donor assignments, decoders and streams",
            "inputs": provenance}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(statistics.to_string(index=False))
    print(descriptives.to_string(index=False))


if __name__ == "__main__":
    run()
