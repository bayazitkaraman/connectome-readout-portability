"""Recompute cohort, donor, reconstruction and displacement summaries for the paper."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nki_numerical_repeat import fetch_data
from paired_pool_augmentation import ROOT
from transfer_donor_checks import donor_summary, prepare


RESULTS = ROOT / "results"
OUT = RESULTS / "reviewer-context"
AGE_HASH = "3ceae2b45965bec28b0669243659a39946d473e9f6acd97d6a5023ded0a14ffc"
FLAGGED = "A00063383"


def bootstrap_indices(n):
    return np.random.default_rng(20261120).integers(n, size=(10000, n))


def interval(draws):
    lo, hi = np.quantile(draws, [.025, .975])
    return {"lo": lo, "hi": hi}


def cohort_context(cohort):
    graphs, ids, scores = prepare(cohort)
    upper = np.triu_indices(graphs.shape[-1], 1)
    edges = graphs[:, :, upper[0], upper[1]]
    counts = np.count_nonzero(edges, axis=-1)
    density = []
    for visit, values in enumerate(counts, 1):
        density.append(dict(cohort=cohort, visit=visit, people=len(ids), possible=edges.shape[-1],
                            mean_edges=values.mean(), median_edges=np.median(values),
                            min_edges=values.min(), max_edges=values.max(),
                            mean_density=values.mean()/edges.shape[-1]))
    own = scores.groupby("target_id").own_similarity.first()
    top_other = scores.groupby("target_id").other_similarity.max()
    winners, original = donor_summary(scores)
    cached = pd.read_csv(RESULTS/"transfer-similarity/nearest_donor_people.csv",
                         dtype={"target_id": str, "selected_id": str}).query("cohort == @cohort")
    np.testing.assert_array_equal(winners.selected_id, cached.selected_id)
    np.testing.assert_allclose(winners.target_blind_nearest_donor_r2, cached.target_blind_nearest_donor_r2)
    frequencies = winners.selected_id.value_counts()
    # Disallow all donors tied at the second largest observed count, even if that leaves >2 out.
    excluded = set(frequencies[frequencies >= frequencies.iloc[1]].index)
    _, sensitivity = donor_summary(scores.loc[~scores.other_id.isin(excluded)])
    summary = dict(cohort=cohort, people=len(ids), closest_is_own=int((own > top_other).sum()),
                   own_tied_for_top=int(np.isclose(own, top_other, rtol=0., atol=1e-12).sum()),
                   flagged_own_similarity=own.get(FLAGGED, np.nan),
                   unique_selected=len(frequencies), top_count=frequencies.iloc[0],
                   second_count=frequencies.iloc[1], selected_r2=original["selected_mean_r2"],
                   target_blind_r2=original["target_blind_r2"], target_aware_gain=original["target_aware_minus_blind"],
                   lo=original["target_aware_minus_blind_ci"][0], hi=original["target_aware_minus_blind_ci"][1],
                   excluded_frequent_donor_count=len(excluded),
                   restricted_target_aware_gain=sensitivity["target_aware_minus_blind"],
                   restricted_lo=sensitivity["target_aware_minus_blind_ci"][0],
                   restricted_hi=sensitivity["target_aware_minus_blind_ci"][1])
    return density, summary


def displacement(cohort):
    people = pd.read_csv(RESULTS/"portability-path/shift_people.csv", dtype={"source_id": str})
    alpha = 10. if cohort == "nki" else 1.
    p = people.query("cohort == @cohort and source_set == 'first_visit' and rung == 'between_visit' "
                     "and task == 'linear' and alpha == @alpha").sort_values("source_id")
    assert len(p) == p.source_id.nunique() == (25 if cohort == "nki" else 57)
    iso = p.expected_isotropic_loss.to_numpy()
    spatial = p.actual_output_shift.to_numpy()
    real = (p.source_r2 - p.target_r2).to_numpy()
    np.testing.assert_allclose(iso, p.source_r2 - p.isotropic_r2, atol=1e-12)
    np.testing.assert_allclose(real, spatial + p.error_cross_term, atol=1e-12)
    extra = real - iso
    directed = spatial - iso
    draws = bootstrap_indices(len(p))
    assert (extra[draws].mean(axis=1) > 0).all()
    rows = []
    for name, value in [("isotropic", iso), ("spatial_second_moment", spatial), ("measured_repeat", real),
                        ("spatial_minus_isotropic", directed), ("repeat_minus_spatial", real - spatial),
                        ("repeat_minus_isotropic", extra)]:
        rows.append(dict(cohort=cohort, n=len(p), alpha=alpha, statistic=name, estimate=value.mean(),
                         **interval(value[draws].mean(axis=1))))
    rows.append(dict(cohort=cohort, n=len(p), alpha=alpha, statistic="fraction_excess_spatial",
                     estimate=directed.mean()/extra.mean(),
                     **interval(directed[draws].mean(axis=1) / extra[draws].mean(axis=1))))
    if cohort == "nki":
        retained = extra[p.source_id != FLAGGED]
        rows.append(dict(cohort=cohort, n=len(retained), alpha=alpha, statistic="repeat_minus_iso_omit_flagged_target",
                         estimate=retained.mean(), **interval(retained[bootstrap_indices(len(retained))].mean(axis=1))))
    return rows


def rerun_context():
    subjects, visits, halves, refs, graphs = fetch_data(ROOT/"data/nki/inputs.h5")
    path = pd.read_csv(RESULTS/"portability-path/rung_pairs.csv.gz")
    current = path.query("task == 'linear' and source_visit == 1 and source_half == 'odd' and "
                         "rung == 'numerical_perturbation' and alpha == .001")
    rows = []
    u, v = np.triu_indices(refs.shape[-1], 1)
    for i, subject in enumerate(subjects):
        reference = refs[0, 0, i][u, v]
        scores = current[current.source_id == subject].sort_values("run")
        np.testing.assert_array_equal(scores.run, np.arange(20))
        runs = [graphs.loc[(subject, visits[0, i], halves[0], f"sim-{k}")][u, v] for k in range(20)]
        same = [np.array_equal(reference, run) for run in runs]
        np.testing.assert_array_equal(same, scores.identical_offdiag)
        rows.append(dict(participant=subject, distinct_rerun_graphs=len(np.unique(runs, axis=0)),
                         unchanged=sum(same), changed=20-sum(same), mean_r2=scores.target_r2.mean(),
                         changed_only_mean_r2=scores.target_r2[~np.array(same)].mean()))
    return rows


def age_context(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == AGE_HASH
    with np.load(ROOT/"data/nki/nki_reference.npz", allow_pickle=False) as archive:
        subjects = archive["subjects"]
        assert set(archive["standard_session_labels"][0]) == {"BAS1"}
    index = pd.read_csv(path, dtype=str)
    baseline = index.loc[index.subject.isin(subjects) & index.session.eq("BAS1")]
    ages = []
    for suffix in ("_dwi.nii.gz", "_T1w.nii.gz"):
        frame = baseline.loc[baseline.filepath.str.endswith(suffix)].set_index("subject")
        assert len(frame) == len(subjects) == frame.index.nunique()
        ages.append(frame.loc[subjects, "age"].astype(float).to_numpy())
    np.testing.assert_array_equal(*ages)
    a = ages[0]
    return dict(cohort="nki", visit="BAS1", n=len(a), minimum=a.min(), maximum=a.max(),
                q1=np.quantile(a, .25), median=np.median(a), q3=np.quantile(a, .75),
                mean=a.mean(), sd=a.std(ddof=1), age_16_to_17=int(np.isin(a, [16,17]).sum()),
                age_19_to_30=int(np.count_nonzero((a>=19) & (a<=30))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nki-age-index", type=Path, help="Checksum-pinned Rockland imaging release index")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    density, donor, shifts = [], [], []
    for cohort in ("nki", "bnu1"):
        cohort_density, selection = cohort_context(cohort)
        density.extend(cohort_density)
        donor.append(selection)
        shifts.extend(displacement(cohort))
    pd.DataFrame(density).to_csv(OUT/"edge_density.csv", index=False)
    pd.DataFrame(donor).to_csv(OUT/"donor_context.csv", index=False)
    pd.DataFrame(shifts).to_csv(OUT/"spatial_decomposition.csv", index=False)
    pd.DataFrame(rerun_context()).to_csv(OUT/"numerical_context.csv", index=False)
    gains = pd.read_csv(RESULTS/"source-augmentation/people.csv").query("arm == 'direction_halves'").gain_r2
    pd.DataFrame([dict(n=len(gains), mean=gains.mean(), max_gain=gains.max(),
                       mean_without_max=gains.drop(gains.idxmax()).mean(),
                       **interval(gains.to_numpy()[bootstrap_indices(len(gains))].mean(axis=1)))]).to_csv(
                           OUT/"direction_half_augmentation.csv", index=False)
    if args.nki_age_index:
        pd.DataFrame([age_context(args.nki_age_index)]).to_csv(OUT/"age_aggregate.csv", index=False)


if __name__ == "__main__":
    main()
