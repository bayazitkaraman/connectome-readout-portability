"""Describe aligned readout states under measured graph replacements."""
import json
from time import perf_counter

import nki_numerical_repeat as numerical
from run_visit_portability_path import OUT, fresh_test, pilot, read_graphs, transfer
import numpy as np
import pandas as pd


METRICS = ["displacement_pct", "pattern_r", "amplitude_ratio", "mean_shift_pct"]
IDENTITY = ["cohort", "rung", "source_id", "target_id", "source_visit", "target_visit",
            "source_half", "target_half", "run"]
BOOTSTRAP_SEED = 20261120


def train_scale(net, mapping, stream):
    states, _ = pilot.simulate(net, mapping, stream, 0, pilot.FIT_N)
    return np.sqrt(states.var(axis=0).mean())


def compare_states(source, target, training_rms):
    """Center within parcel over all test times; then compare aligned time-parcel entries."""
    sm, tm = source.mean(axis=0), target.mean(axis=0)
    sc, tc = source - sm, target - tm
    sp, tp = np.mean(sc**2), np.mean(tc**2)
    dot = np.mean(sc * tc)
    corr = dot / np.sqrt(sp * tp)
    mean_power = np.mean((tm - sm)**2)
    delta_power = np.mean((target - source)**2)
    reconstructed = sp + tp - 2 * dot + mean_power
    relative_error = abs(reconstructed - delta_power) / training_rms**2
    if not np.isfinite(corr) or abs(corr) > 1 + 1e-12 or relative_error > 1e-10:
        raise ValueError("Undefined correlation or failed displacement-energy check")
    return dict(displacement_pct=100*np.sqrt(delta_power)/training_rms,
                pattern_r=np.clip(corr, -1, 1), amplitude_ratio=np.sqrt(tp/sp),
                mean_shift_pct=100*np.sqrt(mean_power)/training_rms,
                train_state_rms=training_rms, source_test_centered_rms=np.sqrt(sp),
                target_test_centered_rms=np.sqrt(tp), displacement_rms=np.sqrt(delta_power),
                reconstruction_error=relative_error)


def visit_rows(cohort):
    graphs, ids, half = read_graphs(cohort)
    n = len(ids)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    targets = [fresh_test(pilot.reservoir(graph, transfer.GAIN), mapping, 10000+n+t)
               for t, graph in enumerate(graphs[1])]
    rows = []
    for d, graph in enumerate(graphs[0]):
        network = pilot.reservoir(graph, transfer.GAIN)
        scale = train_scale(network, mapping, 10000+d)
        for t, (target, labels) in enumerate(targets):
            stream = 10000+n+t
            source, source_labels = fresh_test(network, mapping, stream)
            np.testing.assert_array_equal(source_labels, labels)
            rows.append(dict(cohort=cohort, rung="between_visit" if d == t else "stranger",
                             source_id=ids[d], target_id=ids[t], source_visit=1, target_visit=2,
                             source_half=half, target_half=half, run=-1, source_train_stream=10000+d,
                             test_stream=stream, **compare_states(source, target, scale)))
        if (d+1) % 10 == 0 or d == n-1:
            print(f"{cohort}: completed {d+1}/{n} sources", flush=True)
    return rows


def technical_rows():
    subjects, visits, halves, reference, graphs = numerical.fetch_data(numerical.ROOT/"data/nki/inputs.h5")
    mapping = pilot.input_map(reference.shape[-1], key=10)
    upper = np.triu_indices(reference.shape[-1], 1)
    rows = []
    for i, person in enumerate(subjects):
        source_graph = reference[0, 0, i]
        network = numerical.reservoir(source_graph)
        stream = 10000+i
        scale = train_scale(network, mapping, stream)
        source, source_labels = fresh_test(network, mapping, stream)
        source_key = source_graph[upper].astype("float64").tobytes()
        cache = {source_key: compare_states(source, source, scale)}
        replacements = [("split_half", halves[1], -1, reference[1, 0, i])]
        replacements.extend(("numerical_perturbation", halves[0], run,
                             graphs.loc[(person, visits[0, i], halves[0], f"sim-{run}")])
                            for run in range(20))
        for rung, half, run, graph in replacements:
            key = graph[upper].astype("float64").tobytes()
            if key not in cache:
                target, labels = fresh_test(numerical.reservoir(graph), mapping, stream)
                np.testing.assert_array_equal(source_labels, labels)
                cache[key] = compare_states(source, target, scale)
            rows.append(dict(cohort="nki", rung=rung, source_id=person, target_id=person,
                             source_visit=1, target_visit=1, source_half=halves[0], target_half=half,
                             run=run, source_train_stream=stream, test_stream=stream, **cache[key]))
    return rows


def verify(pairs):
    dtypes = {"source_id": str, "target_id": str}
    visit = pd.read_csv(OUT/"visit_pairs.csv.gz", dtype=dtypes,
                        usecols=IDENTITY[:-1]+["task", "alpha", "source_train_stream", "test_stream"])
    visit = visit.loc[visit.task.eq("linear") & visit.alpha.eq(.001)].assign(run=-1)
    visit_keys = IDENTITY+["source_train_stream", "test_stream"]
    visits = pairs.loc[pairs.target_visit.eq(2)].merge(
        visit[visit_keys], on=visit_keys, how="outer", indicator=True, validate="one_to_one")
    assert len(visits) == 3874 and visits._merge.eq("both").all()
    old = pd.concat([pd.read_csv(OUT/name, dtype=dtypes) for name in
                     ["shift_visit_pairs.csv.gz", "shift_technical_pairs.csv.gz"]], ignore_index=True)
    old = old.loc[old.task.eq("linear") & old.alpha.eq(.001) & old.source_visit.eq(1)
                  & (old.target_visit.eq(2) | old.source_half.eq("odd"))]
    matched = pairs.loc[pairs.rung.ne("stranger")].merge(
        old[IDENTITY+["delta_fraction"]], on=IDENTITY, how="outer", indicator=True, validate="one_to_one")
    assert len(matched) == 607 and matched._merge.eq("both").all()
    err = (matched.displacement_pct - 100*matched.delta_fraction).abs().max()
    np.testing.assert_allclose(matched.displacement_pct, 100*matched.delta_fraction, rtol=1e-11, atol=1e-10)
    assert len(pairs) == 4399 and not pairs.duplicated(IDENTITY).any() and not pairs.isna().any().any()
    return dict(ordered_visit_pairs_checked=len(visits), archived_displacements_checked=len(matched),
                max_archived_error_percentage_points=err,
                max_training_normalized_energy_error=pairs.reconstruction_error.max())


def summarize(pairs):
    people = pairs.groupby(["cohort", "rung", "target_id"])[METRICS].mean()
    people["pairs"] = pairs.groupby(["cohort", "rung", "target_id"]).size()
    summary = people.groupby(["cohort", "rung"])[METRICS].mean()
    summary["people"] = people.groupby(["cohort", "rung"]).size()
    summary["pairs"] = people.groupby(["cohort", "rung"]).pairs.sum()
    contrasts = []
    for cohort, group in people.groupby(level="cohort"):
        own = group.xs("between_visit", level="rung")[METRICS]
        other = group.xs("stranger", level="rung")[METRICS]
        assert own.index.equals(other.index)
        diffs = (own-other).to_numpy()
        draw = np.random.default_rng(BOOTSTRAP_SEED).integers(len(diffs), size=(10000, len(diffs)))
        lo, hi = np.quantile(diffs[draw].mean(axis=1), [.025, .975], axis=0)
        for j, metric in enumerate(METRICS):
            contrasts.append(dict(cohort=cohort, metric=metric, targets=len(diffs),
                                  own_minus_stranger=diffs[:, j].mean(), ci_low=lo[j], ci_high=hi[j]))
    return people.reset_index(), summary.reset_index(), pd.DataFrame(contrasts)


def main():
    start = perf_counter()
    rows = visit_rows("nki") + visit_rows("bnu1") + technical_rows()
    pairs = pd.DataFrame(rows)
    pairs[["source_id", "target_id"]] = pairs[["source_id", "target_id"]].astype(str)
    checks = verify(pairs)
    people, summary, contrasts = summarize(pairs)
    pairs.to_csv(OUT/"trajectory_pairs.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    people.to_csv(OUT/"trajectory_people.csv", index=False)
    summary.to_csv(OUT/"trajectory_summary.csv", index=False)
    contrasts.to_csv(OUT/"trajectory_contrasts.csv", index=False)
    checks.update(
        source_to_target="First to second visit (NKI odd, Beijing all); NKI technical comparisons start from first-visit odd.",
        seed=pilot.SEED, map_key=10, train_role=0, test_role=61, test_streams=pilot.TEST_STREAMS,
        displacement_pct="100 times raw RMS difference on aligned test states divided by the source's centered training RMS.",
        pattern_r="Subtract the temporal mean of each parcel separately in each graph over the concatenated test streams. Compute the cosine of the two flattened centered arrays, retaining parcel and sample alignment.",
        amplitude_ratio="Target centered test RMS divided by source centered test RMS, with each parcel centered over time.",
        mean_shift_pct="RMS across parcels of the difference in their test means, as a percentage of centered source training RMS.",
        aggregation="Average pairs within target and rung, then average target means. Correlations are ordinary arithmetic means of pair-level coefficients.",
        interval="Paired target bootstrap of mean own minus within-target mean stranger, conditional on observed donor panel and streams, 10,000 draws, percentile 95% intervals.",
        bootstrap_seed=BOOTSTRAP_SEED)
    (OUT/"trajectory_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Verified all trajectories in {perf_counter() - start:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
