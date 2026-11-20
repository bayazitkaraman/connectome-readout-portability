"""Fit fixed-budget linear readouts on one or several NKI first-visit reconstructions."""
import json

import nki_numerical_repeat as numerical
from run_visit_portability_path import fresh_test
from unlabeled_parcel_alignment import r_squared
import numpy as np
import pandas as pd


ROOT = numerical.ROOT
OUT = ROOT / "results/source-augmentation"
PILOT = numerical.PILOT
ARMS = ("original", "numerical", "direction_halves")
ALPHA = 10.
ALLOCATION_ROLE = 220
BOOTSTRAP_SEED = 20261120


def balanced_design(arrays, participant_index):
    rows = len(arrays[0])
    count = len(arrays)
    if rows % count:
        raise ValueError("Each source graph must contribute the same number of training times.")
    allocation = PILOT.rng(ALLOCATION_ROLE, participant_index, count).permutation(rows)
    mixed = np.empty_like(arrays[0])
    for states, indices in zip(arrays, np.split(allocation, count)):
        mixed[indices] = states[indices]
    return mixed


def fitted_scores(design, targets, penalty, xs, xt, y):
    fit = PILOT.RidgeFamily.from_design((design, targets))
    weights = fit.weights(penalty / fit.scale)
    source_pred = (xs - fit.mx) @ weights + fit.my
    target_pred = (xt - fit.mx) @ weights + fit.my
    # An independent covariance-form solve checks the fixed absolute penalty.
    xc, yc = design - design.mean(axis=0), targets - targets.mean(axis=0)
    covariance = xc.T @ xc / len(design)
    cross_covariance = xc.T @ yc / len(design)
    direct = np.linalg.solve(covariance + penalty * np.eye(design.shape[1]), cross_covariance)
    error = 0.
    for states, predictions in ((xs, source_pred), (xt, target_pred)):
        checked = (states - design.mean(axis=0)) @ direct + targets.mean(axis=0)
        error = max(error, float(np.max(np.abs(predictions - checked))))
    assert error < 1e-10
    return float(r_squared(y, source_pred)), float(r_squared(y, target_pred)), error


def analyze():
    subjects, visits, halves, ref, variants = numerical.fetch_data(ROOT / "data/nki/inputs.h5")
    mapping = PILOT.input_map(ref.shape[-1], key=10)
    upper = np.triu_indices(ref.shape[-1], 1)
    rows = []
    identity_error = 0.
    solve_error = 0.
    identical_reruns = 0
    for i, subject in enumerate(subjects):
        train_id = 10000 + i
        test_id = train_id + len(subjects)
        raw_source = ref[0, 0, i]
        source_net = numerical.reservoir(raw_source)
        training, train_y = PILOT.simulate(source_net, mapping, train_id, 0, PILOT.FIT_N)
        penalty = ALPHA * float(training.var(axis=0).mean())
        xs, labels = fresh_test(source_net, mapping, test_id)
        xt, repeat_labels = fresh_test(numerical.reservoir(ref[0, 1, i]), mapping, test_id)
        np.testing.assert_array_equal(labels, repeat_labels)
        y = labels[:, :len(PILOT.DELAYS)]
        key = lambda graph: np.ascontiguousarray(graph[upper], dtype=np.float64).tobytes()
        source_key = key(raw_source)
        cache = {source_key: training}

        def training_states(graph):
            graph_key = key(graph)
            if graph_key not in cache:
                states, labels_variant = PILOT.simulate(
                    numerical.reservoir(graph), mapping, train_id, 0, PILOT.FIT_N)
                np.testing.assert_array_equal(train_y, labels_variant)
                cache[graph_key] = states
            return cache[graph_key]

        run_graphs = [variants.loc[(subject, visits[0, i], halves[0], f"sim-{k}")] for k in range(20)]
        identical_reruns += sum(key(raw) == source_key for raw in run_graphs)
        graph_banks = [[raw_source], [raw_source, *run_graphs], [raw_source, ref[1, 0, i]]]
        comparator = None
        for arm, bank in zip(ARMS, graph_banks):
            count = len(bank)
            design = balanced_design([training_states(graph) for graph in bank], i)
            source, target, error = fitted_scores(design, train_y, penalty, xs, xt, y)
            solve_error = max(solve_error, error)
            if arm == "original":
                comparator = (source, target)
            else:
                identity = balanced_design([training] * count, i)
                np.testing.assert_array_equal(identity, training)
                ref_source, ref_repeat, _ = fitted_scores(identity, train_y, penalty, xs, xt, y)
                identity_error = max(identity_error, abs(ref_source-comparator[0]), abs(ref_repeat-comparator[1]))
            rows.append(dict(cohort="nki", participant_id=str(subject), arm=arm, graph_count=count,
                             distinct_graphs=len(set(key(graph) for graph in bank)), training_rows=len(design),
                             rows_per_graph=len(design)//count, source_train_stream=train_id, test_stream=test_id,
                             source_alpha=ALPHA, absolute_lambda=penalty, source_r2=source, repeat_r2=target,
                             gain_r2=target-comparator[1]))
    assert identity_error < 1e-12
    people = pd.DataFrame(rows)
    assert not people.duplicated(["participant_id", "arm"]).any() and len(people) == 75
    ref_records = pd.read_csv(ROOT / "results/portability-path/shift_people.csv", dtype={"source_id": str})
    archived = ref_records.loc[ref_records.cohort.eq("nki") & ref_records.source_set.eq("first_visit")
                               & ref_records.rung.eq("between_visit") & ref_records.task.eq("linear")
                               & ref_records.alpha.eq(ALPHA), ["source_id", "source_r2", "target_r2"]]
    baseline = people[people.arm.eq("original")].merge(
        archived, left_on="participant_id", right_on="source_id", suffixes=("", "_archived"),
        how="outer", validate="one_to_one", indicator=True)
    assert len(baseline) == 25 and baseline._merge.eq("both").all()
    source_error = float((baseline.source_r2 - baseline.source_r2_archived).abs().max())
    repeat_error = float((baseline.repeat_r2 - baseline.target_r2).abs().max())
    assert max(source_error, repeat_error) < 1e-11
    checks = dict(verified_participants=len(baseline), rows=len(people), hdf_md5=numerical.HDF_MD5,
                  numerical_reruns=500, numerical_reruns_identical_to_reference=identical_reruns,
                  row_seed_entropy=[PILOT.SEED, ALLOCATION_ROLE, "participant_index", "graph_count"],
                  train_role=0, test_role=61, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=10000,
                  max_archived_source_r2_error=source_error, max_archived_repeat_r2_error=repeat_error,
                  max_identical_augmentation_r2_error=identity_error, max_direct_prediction_error=solve_error)
    return people, checks


def summarize(people):
    participant_ids = sorted(people.participant_id.unique())
    ref = people[people.arm.eq("original")].set_index("participant_id").loc[participant_ids]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(ref), size=(10000, len(ref)))
    original_ratio = ref.repeat_r2.mean() / ref.source_r2.mean()
    draw_ratio = ref.repeat_r2.to_numpy()[draws].mean(axis=1) / ref.source_r2.to_numpy()[draws].mean(axis=1)
    records = []
    for arm in ARMS:
        group = people[people.arm.eq(arm)].set_index("participant_id").loc[participant_ids]
        source, target, gain = group[["source_r2", "repeat_r2", "gain_r2"]].to_numpy().T
        boot_source, boot_target, boot_gain = [score[draws].mean(axis=1) for score in (source, target, gain)]
        ratio = target.mean() / source.mean()
        boot_ratio = boot_target / boot_source
        record = dict(cohort="nki", arm=arm, n_people=len(source), total_rows=PILOT.FIT_N,
                      graph_count=int(group.graph_count.iloc[0]), n_improved=int((gain>0).sum()),
                      n_repeat_positive=int((target>0).sum()))
        estimates = {"source_r2": (source.mean(), boot_source), "repeat_r2": (target.mean(), boot_target),
                     "gain_r2": (gain.mean(), boot_gain), "retention_pct": (100*ratio, 100*boot_ratio),
                     "retention_gain_pp": (100*(ratio-original_ratio), 100*(boot_ratio-draw_ratio))}
        for name, (estimate, replicates) in estimates.items():
            record[name] = estimate
            record[name+"_lo"], record[name+"_hi"] = np.quantile(replicates, [.025, .975])
        records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    people, checks = analyze()
    table = summarize(people)
    OUT.mkdir(parents=True, exist_ok=True)
    people.to_csv(OUT/"people.csv", index=False)
    table.to_csv(OUT/"summary.csv", index=False)
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(table.to_string(index=False))
