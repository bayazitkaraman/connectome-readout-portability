"""Check parcelwise displacement power on the ten original maps using archived penalties."""
from pathlib import Path
import json
import os
import time

for env_name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[env_name] = "1"

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "results" / "regional-noise"

from headline_robustness import evaluate_pair, fresh_test, pilot, read_graphs, weights
import numpy as np
import pandas as pd

ROOT_RESULTS = STUDY / "results" / "headline-robustness"
SEED = 20261120
N_BOOT = 10_000
METRICS = ["source_r2", "isotropic_loss", "diagonal_loss", "spatial_loss", "repeat_loss",
           "diagonal_minus_isotropic", "spatial_minus_diagonal", "repeat_minus_spatial",
           "repeat_minus_diagonal", "repeat_minus_isotropic", "diagonal_r2"]


def simulate_and_score(cohort, graphs, ids, map_key, archived):
    mapping = pilot.input_map(graphs.shape[-1], map_key)
    results = []
    errors = {}
    for i, participant in enumerate(ids):
        person = str(participant)
        reference = archived.loc[(cohort, map_key, person)]
        alpha = reference.alpha
        source = pilot.reservoir(graphs[0, i], 0.9)
        target = pilot.reservoir(graphs[1, i], 0.9)
        train = pilot.simulate(source, mapping, 10000 + i, 0, pilot.FIT_N)
        frozen = pilot.RidgeFamily.from_design(train)
        eval_stream = 10000 + i + len(ids)
        xs, ys = fresh_test(source, mapping, eval_stream)
        xt, yt = fresh_test(target, mapping, eval_stream)
        np.testing.assert_array_equal(ys, yt)
        y = ys[:, :len(pilot.DELAYS)]
        observed = evaluate_pair(frozen, xs, xt, y, alpha)
        for name, value in observed.items():
            difference = abs(value - reference[name])
            errors[name] = max(errors.get(name, 0.0), float(difference))
            np.testing.assert_allclose(value, reference[name], rtol=3e-12, atol=5e-11)

        # Parcel j has independent zero-mean noise with E[noise_j^2] = E[delta_j^2].
        # Use the uncentered second moment so that mean changes contribute to the power.
        delta = xt - xs
        parcel_power = np.mean(delta**2, axis=0)
        w = weights(frozen, alpha)
        delay_variance = y.var(axis=0)
        diagonal = np.mean((parcel_power @ (w**2)) / delay_variance)
        spatial = observed["projected_loss"]
        isotropic = observed["noise_loss"]
        real = observed["source_r2"] - observed["repeat_r2"]
        # Direct contraction of off-diagonal second moments checks the signed spatial increment.
        second_moment = (delta.T @ delta) / len(delta)
        np.testing.assert_allclose(second_moment.diagonal(), parcel_power, rtol=1e-12, atol=1e-12)
        np.fill_diagonal(second_moment, 0.0)
        off_diagonal = np.einsum("ik,ij,jk->k", w, second_moment, w) / delay_variance
        np.testing.assert_allclose(spatial - diagonal, off_diagonal.mean(), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(real - spatial, observed["residual_cross_term"], rtol=1e-11, atol=1e-12)
        assert np.isfinite(diagonal) and diagonal >= 0
        results.append(dict(cohort=cohort, target_id=person, map_key=map_key, alpha=alpha,
                            source_r2=observed["source_r2"], isotropic_loss=isotropic,
                            diagonal_loss=diagonal, spatial_loss=spatial, repeat_loss=real,
                            diagonal_minus_isotropic=diagonal-isotropic,
                            spatial_minus_diagonal=spatial-diagonal,
                            repeat_minus_spatial=real-spatial, repeat_minus_diagonal=real-diagonal,
                            repeat_minus_isotropic=real-isotropic,
                            diagonal_r2=observed["source_r2"]-diagonal))
    return results, errors


def summarize(frame):
    entries = []
    for cohort, rows in frame.groupby("cohort", sort=False):
        for label, sub in [("ten_maps", rows)] + [(f"map_{k}", g) for k, g in rows.groupby("map_key")]:
            individuals = sub.groupby("target_id", sort=True)[METRICS].mean()
            draws = np.random.default_rng(SEED).integers(len(individuals), size=(N_BOOT, len(individuals)))
            point = individuals.mean()
            boot = pd.DataFrame(individuals.to_numpy()[draws].mean(axis=1), columns=METRICS)
            for name in ["diagonal_minus_isotropic", "spatial_minus_diagonal", "repeat_minus_spatial",
                         "repeat_minus_diagonal"]:
                key = name + "_share_of_excess_pct"
                point[key] = 100 * point[name] / point.repeat_minus_isotropic
                boot[key] = 100 * boot[name] / boot.repeat_minus_isotropic
            for name, value in point.items():
                lo, hi = np.quantile(boot[name], [0.025, 0.975])
                positive = int((individuals[name] > 0).sum()) if name in METRICS else None
                entries.append(dict(cohort=cohort, maps=label, metric=name, estimate=value,
                                    lower=lo, upper=hi, n_people=len(individuals),
                                    n_positive_participant_means=positive))
    return pd.DataFrame(entries)


def main():
    start = time.perf_counter()
    archived = pd.read_csv(ROOT_RESULTS / "people.csv", dtype={"target_id":str})
    archived = archived[archived.setting.str.startswith("map_") & archived.rule.eq("retuned")]
    archived = archived.set_index(["cohort", "map_key", "target_id"], verify_integrity=True)
    assert len(archived) == 820
    all_rows, all_errors = [], {}
    for cohort, count in [("nki", 25), ("bnu1", 57)]:
        graphs, ids, _ = read_graphs(cohort)
        assert len(ids) == count
        for key in range(10, 20):
            rows, errors = simulate_and_score(cohort, graphs, ids, key, archived)
            all_rows.extend(rows)
            for name, value in errors.items():
                all_errors[name] = max(all_errors.get(name, 0.0), value)
            print(cohort, key, f"{time.perf_counter()-start:.1f}s elapsed", flush=True)
    frame = pd.DataFrame(all_rows)
    assert len(frame) == len(archived)
    summary = summarize(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "participant_map.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    meta = dict(total_rows=len(frame), n_people=82, n_maps=10, seed=SEED,
                draws=N_BOOT, tags=["native", "forward", "retuned", "linear recall"],
                max_difference_from_archived=all_errors,
                note="Intervals are paired percentiles over people, conditional on fixed maps, streams and selected penalties.")
    (OUT / "validation.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(summary.query("maps == 'ten_maps'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
