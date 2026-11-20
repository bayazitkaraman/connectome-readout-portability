"""Probe readout amplitude and the shared comparison-person explanation for anatomy/accuracy associations."""
import json
from pathlib import Path

from paired_pool_augmentation import read_graphs, pilot, ROOT
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


OUT = ROOT / "results/density-controls/mechanisms"
SEED = 620624
REPS = 10_000


def rank_correlations(x, y):
    """One Spearman correlation per row, re-ranking any repeated observations."""
    x_rank = rankdata(x, axis=-1)
    y_rank = rankdata(y, axis=-1)
    x_rank -= x_rank.mean(axis=-1, keepdims=True)
    y_rank -= y_rank.mean(axis=-1, keepdims=True)
    return (x_rank * y_rank).sum(axis=-1) / np.sqrt((x_rank**2).sum(axis=-1) * (y_rank**2).sum(axis=-1))


def record(label, measured, bootstrap, cohort, rule):
    low, high = np.nanquantile(bootstrap, [.025, .975])
    return dict(cohort=cohort, mix_rule=rule, comparison=label, rho=float(measured),
                low=float(low), high=float(high), n_finite=int(np.isfinite(bootstrap).sum()))


def probe_scans(cohort):
    graphs, ids, _ = read_graphs(cohort)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    a, b = np.triu_indices(graphs.shape[-1], 1)
    rows = []
    for visit, matrices in enumerate(graphs):
        for subject, matrix in zip(ids, matrices):
            net = pilot.reservoir(matrix, .90)
            x, _ = pilot.simulate(net, mapping, 29000, 80, pilot.FIT_N)
            rms = np.sqrt(np.mean(x*x))
            assert np.isfinite(rms) and rms > 0
            rows.append(dict(cohort=cohort, subject=subject, visit=visit+1,
                             density=np.count_nonzero(matrix[a, b])/len(a), readout_rms=rms))
    return pd.DataFrame(rows)


def donor_result_sets(cohort, rule, pairs):
    axis = ["subject", "candidate_draw"]
    g = pairs.set_index(axis).sort_index()
    x = g.anatomy_advantage.unstack().to_numpy()
    y = g.accuracy_advantage.unstack().to_numpy()
    n, k = x.shape
    assert k == 3 and not g.index.has_duplicates
    rng = np.random.default_rng(SEED)
    draws = rng.integers(n, size=(REPS,n))
    indices = rng.integers(k, size=(REPS,n,k))
    xmean, ymean = x.mean(axis=1), y.mean(axis=1)
    standardized = spearmanr(xmean, ymean).statistic
    original = rank_correlations(xmean[draws], ymean[draws])
    outputs = [record("same_three_donors_subject_only", standardized, original, cohort, rule)]
    # Preserve anatomy and accuracy pairing on each draw. This is conditional
    # on the three sampled people and does not estimate a full donor population.
    xr = np.take_along_axis(x[draws], indices, axis=2).mean(axis=2)
    yr = np.take_along_axis(y[draws], indices, axis=2).mean(axis=2)
    outputs.append(record("same_three_donors_subject_and_within_target_donors", standardized,
                          rank_correlations(xr, yr), cohort, rule))
    fold_stats, fold_draws = [], []
    for j in range(k):
        y_other = y[:, np.arange(k) != j].mean(axis=1)
        fold_stats.append(spearmanr(x[:,j], y_other).statistic)
        fold_draws.append(rank_correlations(x[:,j][draws], y_other[draws]))
    outputs.append(record("disjoint_donors_mean_of_three_splits", np.mean(fold_stats),
                          np.mean(fold_draws,axis=0), cohort, rule))
    per_person = pairs.groupby("subject", sort=True).mean(numeric_only=True)
    assert len(per_person) == n
    for u,v in [("density_advantage","amplitude_advantage"),
                ("amplitude_advantage","accuracy_advantage")]:
        uvals, vvals = per_person[u].to_numpy(), per_person[v].to_numpy()
        outputs.append(record(f"{u}_vs_{v}", spearmanr(uvals,vvals).statistic,
                              rank_correlations(uvals[draws],vvals[draws]),cohort,rule))
    return outputs


def add_mismatches(pairs, scans):
    means = scans.pivot(index="subject", columns="visit", values=["density","readout_rms"])
    pairs = pairs.copy()
    for name, col in [("density", "density"), ("amplitude", "readout_rms")]:
        first, second = means[(col,1)], means[(col,2)]
        if name == "amplitude":
            first, second = np.log(first), np.log(second)
        target = pairs.subject.map(second)
        pairs[name+"_advantage"] = ((target-pairs.other_subject.map(first)).abs()
                                    - (target-pairs.subject.map(first)).abs())
    assert pairs[["density_advantage","amplitude_advantage"]].notna().all().all()
    return pairs


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    pairs = pd.read_csv(ROOT / "results/anatomy-advantage/pairs.csv",
                        dtype={"subject":str,"other_subject":str})
    scans, tables, all_stats = [], [], []
    for cohort in ("nki","bnu1"):
        data = probe_scans(cohort)
        scans.append(data)
        for visit, group in data.groupby("visit"):
            x, y = group.density.to_numpy(), group.readout_rms.to_numpy()
            draws = np.random.default_rng(SEED).integers(len(x), size=(REPS,len(x)))
            all_stats.append(record(f"density_vs_rms_visit_{visit}", spearmanr(x,y).statistic,
                                    rank_correlations(x[draws], y[draws]), cohort, "all"))
        compared = add_mismatches(pairs[pairs.cohort == cohort], data)
        tables.append(compared)
        for rule, group in compared.groupby("mix_rule"):
            all_stats.extend(donor_result_sets(cohort, rule, group))
    pd.concat(scans).to_csv(OUT/"scan_amplitudes.csv", index=False)
    pd.concat(tables).to_csv(OUT/"pair_amplitude_mismatches.csv", index=False)
    pd.DataFrame(all_stats).to_csv(OUT/"correlations.csv", index=False)
    (OUT/"settings.json").write_text(json.dumps({"fixed_gain":.9, "input_map_key":10, "ident":29000,
           "role":80,"after_warmup_steps":2352,"resamples":REPS,"seed":SEED},indent=2)+"\n")


if __name__ == "__main__":
    run()
