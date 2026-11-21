"""Verify the NKI visit crosswalk and repeat the ten-map comparison after one exclusion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_nki import CROSSWALK_SHA, MD5, digest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/nki"
HEADLINE = ROOT / "results/headline-robustness"
OUT = ROOT / "results/nki-visit-sensitivity"
UNMATCHED = "A00034854"
LOW_SIMILARITY = "A00063383"
MAPS = [f"map_{key}" for key in range(10, 20)]
METRICS = ["source_r2", "noise_r2", "repeat_r2", "excess_r2", "shift_pct"]
SEED = 20261120
DRAWS = 10000


def verify_source_visits():
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    hdf, crosswalk, prepared = (SOURCE / name for name in ("inputs.h5", "crosswalk.csv", "nki_reference.npz"))
    hashes = dict(source_hdf_md5=digest(hdf, "md5"), source_hdf_sha256=digest(hdf, "sha256"),
                  visit_crosswalk_sha256=digest(crosswalk, "sha256"), prepared_npz_sha256=digest(prepared, "sha256"))
    assert hashes["source_hdf_md5"] == MD5 == provenance["md5"]
    assert hashes["source_hdf_sha256"] == provenance["source_sha256"]
    assert hashes["visit_crosswalk_sha256"] == CROSSWALK_SHA == provenance["crosswalk_sha256"]
    # NPZ container bytes vary with compression. Check its contents against the source below.
    hashes["recorded_npz_sha256"] = provenance["npz_sha256"]
    hashes["prepared_npz_matches_recorded_file"] = hashes["prepared_npz_sha256"] == provenance["npz_sha256"]
    hdf_data = pd.read_hdf(hdf, key="graphs")
    graphs = hdf_data.query("simulation == 'ref' and pipeline == 'det'").set_index(
        ["subject", "session", "directions"], verify_integrity=True)
    assert len(graphs) == 100
    links = pd.read_csv(crosswalk)[["A_number", "old_session_name", "new_session_id"]].drop_duplicates()
    links = links[links.A_number.isin(graphs.index.get_level_values("subject"))]
    lookup = links.set_index(["A_number", "old_session_name"], verify_integrity=True).new_session_id
    visit_rows = []
    with np.load(prepared, allow_pickle=False) as archive:
        weights = archive["weights"]
        assert weights.shape == (2, 2, 25, 83, 83) and list(archive["halves"]) == ["odd", "even"]
        np.testing.assert_array_equal(archive["subjects"], sorted(graphs.index.get_level_values("subject").unique()))
        assert archive["session_labels"].shape == archive["standard_session_labels"].shape == (2, 25)
        assert np.all(archive["standard_session_labels"][0] == "BAS1")
        assert np.all(archive["standard_session_labels"][1] != "BAS1")
        hashes["prepared_weights_sha256"] = hashlib.sha256(np.asarray(weights, dtype="<f8").tobytes()).hexdigest()
        assert hashes["prepared_weights_sha256"] == provenance["weights_sha256"]
        u, v = np.triu_indices(weights.shape[-1], 1)
        for i, person in enumerate(archive["subjects"]):
            odd = weights[0, :, i]
            similarity = np.corrcoef(odd[0, u, v], odd[1, u, v])[0, 1]
            for visit in range(2):
                legacy = str(archive["session_labels"][visit, i])
                standard = str(archive["standard_session_labels"][visit, i])
                assert lookup.get((person, legacy), "unmapped") == standard
                for half, name in enumerate(archive["halves"]):
                    np.testing.assert_array_equal(weights[half, visit, i], graphs.loc[(person, legacy, name), "graph"])
                visit_rows.append(dict(target_id=person, visit=visit+1, legacy_session=legacy,
                                       crosswalk_session=standard, source_halves_verified=2,
                                       odd_visit_pair_edge_correlation=similarity))
    visits = pd.DataFrame(visit_rows)
    missing = visits.loc[visits.crosswalk_session.eq("unmapped"), ["target_id", "legacy_session"]]
    assert list(missing.itertuples(index=False, name=None)) == [(UNMATCHED, "NFB2R")]
    return visits, dict(**hashes, checked_source_matrices=len(graphs), crosswalk_visit_rows=len(visits),
                        visit_pairs=25, excluded_id=UNMATCHED, low_similarity_id=LOW_SIMILARITY)


def recheck_penalties():
    people = pd.read_csv(HEADLINE / "people.csv", dtype={"target_id": str})
    people = people.query("cohort == 'nki' and rule == 'retuned' and setting in @MAPS").copy()
    losses = pd.read_csv(HEADLINE / "validation.csv.gz", dtype={"target_id": str})
    losses = losses.query("cohort == 'nki' and setting in @MAPS")
    assert len(people) == 250 and len(losses) == 4750
    keys = ["setting", "target_id"]
    checked = []
    for setting, group in losses.groupby("setting"):
        wide = group.pivot(index="target_id", columns="alpha", values="validation_mse").sort_index(axis=1)
        assert wide.shape == (25, 19)
        for target in wide.index:
            full_alpha = wide.drop(index=target).mean().idxmin()
            reduced_alpha = np.nan
            if target != UNMATCHED:
                reduced_alpha = wide.drop(index=[target, UNMATCHED]).mean().idxmin()
            checked.append(dict(setting=setting, target_id=target,
                                all_25_alpha=full_alpha, retained_24_alpha=reduced_alpha))
    people = people.merge(pd.DataFrame(checked), on=keys, validate="one_to_one")
    np.testing.assert_array_equal(people.alpha, people.all_25_alpha)
    included = people.target_id != UNMATCHED
    np.testing.assert_array_equal(people.loc[included, "alpha"], people.loc[included, "retained_24_alpha"])
    np.testing.assert_allclose(people.noise_r2 - people.repeat_r2, people.excess_r2, atol=1e-12)
    people["included_after_exclusion"] = included
    return people[keys + ["all_25_alpha", "retained_24_alpha", "included_after_exclusion"] + METRICS]


def summarize(rows):
    # Sort people for identical participant draws across every setting of a given sample.
    people = rows.groupby("target_id", sort=True)[METRICS].mean()
    draws = np.random.default_rng(SEED).integers(len(people), size=(DRAWS, len(people)))
    excess = people.excess_r2.to_numpy()
    low, high = np.quantile(excess[draws].mean(axis=1), [.025, .975])
    return dict(n_people=len(people), **people.mean().to_dict(), excess_r2_low=low,
                excess_r2_high=high, positive_excess=int((excess > 0).sum()))


def main():
    visits, verification = verify_source_visits()
    people = recheck_penalties()
    assert set(people.target_id) == set(visits.target_id)
    participant_means = people.groupby(["target_id", "included_after_exclusion"], as_index=False)[METRICS].mean()
    summaries, per_map = [], []
    for sample, rows in (("all_25", people), ("mapped_24", people[people.included_after_exclusion])):
        summaries.append(dict(sample=sample, **summarize(rows)))
        for setting, group in rows.groupby("setting"):
            per_map.append(dict(sample=sample, setting=setting, **summarize(group)))
    full = pd.read_csv(HEADLINE / "map_portability.csv")
    full = full.query("cohort == 'nki' and setting == 'mean_ten_maps'").iloc[0]
    cols = METRICS + ["excess_r2_low", "excess_r2_high"]
    reproduced = np.array([summaries[0][name] for name in cols])
    expected = full[cols].to_numpy(dtype=float)
    np.testing.assert_allclose(reproduced, expected, rtol=1e-12, atol=1e-12)
    verification.update(archived_choices_verified=250, exclusion_choices_verified=240,
                        exclusion_tuning_people_per_target=23, changed_choices=0,
                        reference_error_per_metric=dict(zip(cols, np.abs(reproduced-expected))),
                        participant_bootstrap_draws=DRAWS, bootstrap_seed=SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    visits.to_csv(OUT / "source_visits.csv", index=False)
    people.to_csv(OUT / "participant_map.csv", index=False)
    participant_means.to_csv(OUT / "participant_means.csv", index=False)
    pd.DataFrame(per_map).to_csv(OUT / "map_summary.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries)[["sample", "excess_r2", "excess_r2_low", "excess_r2_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
