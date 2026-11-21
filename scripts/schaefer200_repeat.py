"""Evaluate Beijing first-to-repeat linear recall on the archived Schaefer-200 input map."""
import hashlib
import json
import os
from pathlib import Path
import re

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

from headline_robustness import evaluate_case, GRID, pilot, summarize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATLAS_RESULTS = ROOT / "results" / "pipeline-robustness"
DATA = ROOT / "data" / "replication_candidates" / "corr"
OUT = ROOT / "results" / "schaefer200-repeat"


def read_inputs():
    manifest = json.loads((ATLAS_RESULTS / "schaefer200_manifest.json").read_text())
    with np.load(DATA / "bnu1_schaefer200.npz", allow_pickle=False) as archive:
        graphs, people, sessions = archive["weights"], archive["subjects"], archive["sessions"]
    with np.load(DATA / "bnu1_dkt_retest.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(people, archive["subjects"])
        np.testing.assert_array_equal(sessions, archive["sessions"])
    assert graphs.shape == (2, 57, 200, 200) and list(sessions) == ["1", "2"]
    np.testing.assert_array_equal(graphs, graphs.swapaxes(-1, -2))
    assert (graphs >= 0).all() and (np.diagonal(graphs, axis1=-2, axis2=-1) == 0).all()
    weights_hash = hashlib.sha256(graphs.tobytes()).hexdigest()
    assert weights_hash == manifest["weights_sha256"]
    expected_ids = {(person, session) for person in people for session in sessions}
    manifest_ids = {re.search(r"sub-(\d+)_ses-(\d+)", row["key"]).groups() for row in manifest["files"]}
    assert manifest_ids == expected_ids and len(manifest["files"]) == len(expected_ids)
    sites = pd.read_csv(ATLAS_RESULTS / "scale_input_sites.csv")
    mapping, readout = pilot.input_map(200, 10)
    np.testing.assert_array_equal(np.flatnonzero(mapping[0]) + 1, sites.parcel_200)
    np.testing.assert_array_equal(mapping[0, sites.parcel_200 - 1], sites.input_weight)
    assert len(readout) == 192
    return graphs, people, sites, weights_hash


def verify_scores(rows, validation, people):
    selected = rows.loc[rows.rule.eq("retuned")].set_index("target_id")
    assert len(selected) == 57 and selected.index.is_unique
    wide = validation.pivot(index="target_id", columns="alpha", values="validation_mse").loc[people, GRID]
    independent_choices = [wide.drop(index=target).mean().idxmin() for target in people]
    np.testing.assert_array_equal(selected.loc[people].alpha, independent_choices)

    old_scores = pd.read_csv(ATLAS_RESULTS / "participant_scores.csv", dtype={"target": str})
    old_own = old_scores.query("scenario == 'schaefer_200' and cohort == 'bnu1'").set_index("target")
    fixed = rows.loc[rows.rule.eq("native_fixed")].set_index("target_id").loc[people, "repeat_r2"]
    repeat_difference = fixed - old_own.loc[people, "own_r2"]
    np.testing.assert_allclose(fixed, old_own.loc[people, "own_r2"], atol=1e-9, rtol=1e-9)
    old_validation = pd.read_csv(ATLAS_RESULTS / "retuned_validation.csv", dtype={"person": str})
    old_validation = old_validation.query("scenario == 'schaefer_200'").pivot(
        index="person", columns="relative_penalty", values="validation_nmse")
    archived_mse = old_validation.loc[people, GRID].to_numpy() * pilot.SIGNAL_VAR
    new_mse = wide.to_numpy()
    np.testing.assert_allclose(new_mse, archived_mse, atol=1e-8, rtol=1e-7)
    differences = np.abs(new_mse - archived_mse) / np.maximum(1, np.abs(archived_mse))
    return dict(all_57_loo_choices_verified=True, n_validation_scores=int(new_mse.size),
                fixed_repeat_max_abs_error=float(repeat_difference.abs().max()),
                validation_max_scaled_error=float(differences.max()),
                selected_counts=selected.alpha.value_counts().to_dict())


def main():
    graphs, people, sites, weights_hash = read_inputs()
    rows, validation = evaluate_case("bnu1", people, "schaefer_200", "schaefer_200", 10, .90, graphs)
    verification = verify_scores(rows, validation, people)
    estimates = {rule: summarize(group) for rule, group in rows.groupby("rule", sort=False)}
    primary = estimates["retuned"]
    details = dict(weights_sha256=weights_hash, input_parcels_one_based=sites.parcel_200.tolist(),
                   input_weights=sites.input_weight.tolist(), **verification,
                   retention_percent=100 * primary["repeat_r2"] / primary["source_r2"],
                   primary=primary)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "people.csv", index=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    pd.DataFrame.from_dict(estimates, orient="index").rename_axis("rule").to_csv(OUT / "summary.csv")
    (OUT / "summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"Noise minus repeat R² = {primary['excess_r2']:.6f} "
          f"[{primary['excess_r2_low']:.6f}, {primary['excess_r2_high']:.6f}]")


if __name__ == "__main__":
    main()
