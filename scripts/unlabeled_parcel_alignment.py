"""Evaluate one fixed, unlabeled parcelwise affine correction on own-repeat linear recall."""
import json
from time import perf_counter

from run_visit_portability_path import fresh_test, pilot, read_graphs, transfer
import numpy as np
import pandas as pd


OUT = transfer.ROOT / "results/portability-baseline"
CALIBRATION_N = 2352
CALIBRATION_ROLE = 219
BOOTSTRAP_SEED = 20261120
PENALTIES = {"nki": 10., "bnu1": 1.}


def moment_map(source_training, target_calibration):
    source_mean = source_training.mean(axis=0)
    target_mean = target_calibration.mean(axis=0)
    source_sd = source_training.std(axis=0)
    target_sd = target_calibration.std(axis=0)
    constant_source = np.ptp(source_training, axis=0) == 0
    constant_target = np.ptp(target_calibration, axis=0) == 0
    varied = ~(constant_source | constant_target)
    ratio = np.divide(source_sd, target_sd, out=np.zeros_like(source_sd), where=varied)
    assert np.isfinite(ratio).all()
    calibr_aligned = source_mean + ratio * (target_calibration - target_mean)
    np.testing.assert_allclose(calibr_aligned.mean(axis=0), source_mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(calibr_aligned.std(axis=0)[varied], source_sd[varied],
                               rtol=1e-9, atol=1e-12)
    return source_mean, target_mean, ratio, constant_source, constant_target


def r_squared(labels, predicted):
    return (1 - np.mean((labels - predicted)**2, axis=0) / labels.var(axis=0)).mean()


def evaluate(cohort):
    graphs, people, half = read_graphs(cohort)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    rows = []
    for i, person in enumerate(people):
        source_stream = 10000 + i
        target_stream = source_stream + len(people)
        source_net, target_net = [pilot.reservoir(visit[i], transfer.GAIN) for visit in graphs]
        xs_train, y_train = pilot.simulate(source_net, mapping, source_stream, 0, pilot.FIT_N)
        model = pilot.RidgeFamily.from_design((xs_train, y_train))
        weights = model.weights(PENALTIES[cohort])

        # Only states leave the dedicated calibration stream.
        calibration = pilot.simulate(target_net, mapping, target_stream, CALIBRATION_ROLE, CALIBRATION_N)[0]
        assert len(calibration) == CALIBRATION_N
        source_mean, target_mean, ratio, constant_s, constant_t = moment_map(xs_train, calibration)

        xs, source_labels = fresh_test(source_net, mapping, target_stream)
        xt, target_labels = fresh_test(target_net, mapping, target_stream)
        np.testing.assert_array_equal(source_labels, target_labels)
        labels = target_labels[:, :len(pilot.DELAYS)]
        assert len(labels) == pilot.TEST_N * pilot.TEST_STREAMS
        aligned = source_mean + (xt - target_mean) * ratio
        source, untouched, corrected = [r_squared(labels, (states-model.mx) @ weights + model.my)
                                        for states in (xs, xt, aligned)]
        rows.append(dict(cohort=cohort, target_id=str(person), source_visit=1, target_visit=2,
                         scan_half=half, alpha=PENALTIES[cohort], source_train_stream=source_stream,
                         target_stream=target_stream, target_calibration_role=CALIBRATION_ROLE,
                         unlabeled_target_rows=CALIBRATION_N, n_readout_parcels=len(ratio),
                         constant_source_parcels=int(constant_s.sum()), constant_target_parcels=int(constant_t.sum()),
                         source_r2=source, untouched_r2=untouched, aligned_r2=corrected, delta_r2=corrected-untouched))
    return rows


def verify(people):
    archive = pd.read_csv(transfer.ROOT/"results/portability-path/visit_pairs.csv.gz",
                          dtype={"target_id": str, "source_id": str},
                          usecols=["cohort", "task", "rung", "target_id", "source_id", "alpha",
                                   "source_r2", "target_r2", "source_train_stream", "test_stream"])
    archive = archive.loc[archive.rung.eq("between_visit") & archive.task.eq("linear")
                          & archive.alpha.eq(archive.cohort.map(PENALTIES))]
    assert archive.target_id.eq(archive.source_id).all()
    keys = ["cohort", "target_id", "alpha", "source_train_stream", "target_stream"]
    archive = archive.rename(columns={"source_r2":"archived_source_r2", "test_stream":"target_stream"})
    merged = people.merge(archive, on=keys, how="outer", indicator=True, validate="one_to_one")
    assert len(merged) == 82 and merged._merge.eq("both").all()
    np.testing.assert_allclose(merged.source_r2, merged.archived_source_r2, rtol=0, atol=1e-11)
    np.testing.assert_allclose(merged.untouched_r2, merged.target_r2, rtol=0, atol=1e-11)
    assert not people.isna().any().any() and np.isfinite(people.aligned_r2).all()
    return dict(verified_people=len(merged), seed=pilot.SEED, calibration_role=CALIBRATION_ROLE,
                calibration_rows=CALIBRATION_N, test_role=61, source_training_role=0,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_samples=10000,
                max_archived_source_error=(merged.source_r2-merged.archived_source_r2).abs().max(),
                max_archived_untouched_error=(merged.untouched_r2-merged.target_r2).abs().max(),
                source_constant_count=int(people.constant_source_parcels.sum()),
                target_constant_count=int(people.constant_target_parcels.sum()))


def summarize(people):
    rows = []
    for cohort, group in people.groupby("cohort", sort=False):
        group = group.sort_values("target_id")
        cols = ["source_r2", "untouched_r2", "aligned_r2", "delta_r2"]
        scores = group[cols].to_numpy()
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(group), size=(10000, len(group)))
        lo, hi = np.quantile(scores[:, 3][draws].mean(axis=1), [.025, .975])
        source, untouched, aligned, delta = scores.mean(axis=0)
        rows.append(dict(cohort=cohort, alpha=PENALTIES[cohort], n_people=len(group),
                         source_r2=source, untouched_r2=untouched, aligned_r2=aligned, delta_r2=delta,
                         delta_lo=lo, delta_hi=hi, untouched_retention=untouched/source,
                         aligned_retention=aligned/source, n_untouched_positive=int((scores[:, 1]>0).sum()),
                         n_aligned_positive=int((scores[:, 2]>0).sum()), n_improved=int((scores[:, 3]>0).sum())))
    return pd.DataFrame(rows)


def main():
    start = perf_counter()
    people = pd.DataFrame(evaluate("nki") + evaluate("bnu1"))
    checks = verify(people)
    summary = summarize(people)
    OUT.mkdir(exist_ok=True, parents=True)
    people.to_csv(OUT/"people.csv", index=False)
    summary.to_csv(OUT/"summary.csv", index=False)
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Verified in {perf_counter()-start:.1f} seconds.")


if __name__ == "__main__":
    main()
