"""Check the collaborator's quantitative claims against stored R² and structural matrices."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from paired_pool_augmentation import read_graphs


RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 20261119


def rank_ci(x, y):
    rng = np.random.default_rng(SEED)
    ix = rng.integers(len(x), size=(10000, len(x)))
    a, b = rankdata(x[ix], axis=1), rankdata(y[ix], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    boot = (a*b).sum(axis=1) / np.sqrt((a*a).sum(axis=1) * (b*b).sum(axis=1))
    return {"rho": float(spearmanr(x, y).statistic), "ci": np.quantile(boot, [.025,.975]).tolist()}


def scan_audit(cohort):
    raw, ids, half = read_graphs(cohort)
    iu = np.triu_indices(raw.shape[-1], k=1)
    e = raw[:, :, iu[0], iu[1]].astype(float)
    e -= e.mean(axis=2, keepdims=True)
    e /= np.linalg.norm(e, axis=2, keepdims=True)
    similarity = e[1] @ e[0].T
    own = np.diag(similarity)
    others = (similarity.sum(axis=1)-own) / (len(ids)-1)
    prev = pd.read_csv(RESULTS / "anatomy-advantage/participants.csv", dtype={"subject":str})
    prev = prev.query("cohort == @cohort and mix_rule == 'equal_donor'").set_index("subject").loc[ids]
    np.testing.assert_allclose(own, prev.own_similarity, atol=1e-12)
    f = pd.read_csv(RESULTS / "density-controls/frozen/single_donor_participant_means.csv",
                    dtype={"target_id":str})
    f = f.query("cohort == @cohort and source_input == 'original_distinct_inputs'").set_index("target_id").loc[ids]
    gain = f.gain.to_numpy()
    split = {}
    for name, x in (("own_similarity", own), ("own_minus_all_other_similarity", own-others),
                    ("own_minus_three_other_similarity", prev.anatomy_advantage.to_numpy())):
        upper = x >= np.median(x)
        split[name] = {"median_similarity":float(np.median(x)), "upper_n":int(upper.sum()),
                       "lower_n":int((~upper).sum()), "upper_gain":float(gain[upper].mean()),
                       "lower_gain":float(gain[~upper].mean()), **rank_ci(x,gain)}
    eight_gains = prev.accuracy_advantage.sort_values(ascending=False)
    return {
        "n":len(ids), "half":half, "all_other_mean_own_lower":int((own < others).sum()),
        "assigned_three_mean_own_lower":int((own < prev.other_similarity).sum()),
        "own_similarity_lowest": float(own.min()), "all_other_similarity_avg_of_lowest":float(others[own.argmin()]),
        "lowest_subject": str(ids[own.argmin()]), "mean_single_own":float(f.own_r2.mean()),
        "mean_single_other":float(f.other_r2.mean()), "single_gain_with_anatomy":split,
        "eight_equal_pool_gain":float(eight_gains.mean()),
        "eight_equal_pool_minus_two_largest":float(eight_gains.iloc[2:].mean()),
        "eight_equal_pool_median":float(eight_gains.median()),
        "two_largest_gains":eight_gains.iloc[:2].tolist(),
        "eight_zero_correlation":pd.read_csv(RESULTS/"density-controls/observational/budget_curves.csv")
            .query("cohort == @cohort and mix_rule == 'equal_donor' and budget in ['0','8']")
            [["budget","spearman"]].to_dict("records"),
    }


def decay_audit(cohort, half):
    folder = RESULTS / f"{cohort}-augmentation"
    def read(path):
        frame = pd.read_csv(path, dtype={"subject":str, "other_subject":str})
        return frame.query("mix_rule == 'equal_donor'")
    coarse = read(folder/"scores.csv")
    refined = read(folder/"refinement/scores.csv")
    df = pd.concat([coarse, refined[~refined.budget.isin(coarse.budget.unique())]])
    eight_check = df[df.budget.isin([0,8])].groupby(["subject","method","budget"]).r2.mean().unstack("budget")
    eight_change = eight_check[8] - eight_check[0]
    eight_changes = {}
    for method, change in eight_change.groupby(level="method"):
        eight_changes[method] = {"n":len(change), "n_improved":int((change>1e-12).sum()),
                                 "n_worse":int((change< -1e-12).sum()), "mean_change":float(change.mean())}
    records = {}
    for k in sorted({0, 8, 16, 32, 64, 48 if cohort == 'nki' else 60}):
        b = df[df.budget == k]
        wide = b.pivot(index=["subject", "other_subject", "repeat"], columns="method", values="r2")
        initial = wide.pool_plus_own - wide.pool_plus_other
        paired = wide.groupby(level=[0,1]).mean()
        delta = paired.pool_plus_own - paired.pool_plus_other
        records[k] = {
            "means": wide.mean().to_dict(), "own_minus_other_mean":float(initial.mean()),
            "max_abs_donor_pair_after_averaging_six":float(delta.abs().max()),
            "max_abs_pair_draw":float(initial.abs().max()),
            "max_abs_any_arm_pair_draw":float((wide.max(axis=1)-wide.min(axis=1)).max()),
            "nonzero_own_other_draws":int((initial!=0).sum()), "total_pair_draws":len(initial),
            "n_draws_keep_by_method":b[b.alpha.eq("keep")].method.value_counts().to_dict(),
        }
    individual = pd.read_csv(RESULTS / f"{cohort}-calibration/scores.csv", dtype={"subject":str})
    individual = individual.query("half == @half and source_visit == 1 and target_visit == 2")
    history = individual[individual.budget.isin([0,8,32,2352])].groupby(["budget","method"]).r2.mean().unstack()
    return {"augmented_means":records,"augmentation_eight_minus_zero":eight_changes,
            "personal_pooled_target_only_means":history.to_dict("index")}


def main():
    report = {}
    for cohort in ("nki","bnu1"):
        a = scan_audit(cohort)
        report[cohort] = {**a, **decay_audit(cohort, a["half"])}
    control = json.loads((RESULTS / "density-controls/positive-control/summary.json").read_text())
    report["synthetic"] = [{k:r[k] for k in ("injection","source_rule","crossings","gain_R2_at_0")} for r in control]
    dest = RESULTS/"nki-split-half"
    dest.mkdir(exist_ok=True)
    (dest/"critique_checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
