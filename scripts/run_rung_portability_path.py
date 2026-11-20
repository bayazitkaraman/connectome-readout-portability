"""Score frozen readouts on paired NKI reference, perturbation and split-half graphs."""
import argparse
import json
from pathlib import Path

import nki_numerical_repeat as numerical
import numpy as np
import pandas as pd


pilot = numerical.PILOT
ROOT = numerical.ROOT
OUT = ROOT / "results" / "portability-path"
PENALTIES = np.r_[numerical.transfer.PENALTIES, 10. ** np.arange(5, 9), np.inf]
TASKS = ("linear", "nonlinear")


def with_product(delay_labels):
    return np.column_stack([delay_labels, delay_labels[:, :3].prod(axis=1)])


def training_path(raw, mapping, ident):
    net = numerical.reservoir(raw)
    x, delayed = pilot.simulate(net, mapping, ident, 0, pilot.FIT_N)
    model = pilot.RidgeFamily.from_design((x, with_product(delayed)))
    # The shared design resolves each delay and the product as independent outputs.
    slopes = np.column_stack([model.weights(a) for a in PENALTIES[:-1]])
    return model, slopes


def test_path(raw, mapping, ident, model, slopes):
    net = numerical.reservoir(raw)
    x, delayed = pilot.combine([
        pilot.simulate(net, mapping, ident, numerical.ROLE, pilot.TEST_N, k)
        for k in range(pilot.TEST_STREAMS)
    ])
    y = with_product(delayed)
    centered = y - model.my
    # Direct predictions avoid subtracting large nearly equal moment terms.
    predictions = ((x - model.mx) @ slopes).reshape(len(y), len(PENALTIES)-1, y.shape[1])
    errors = np.mean(np.square(predictions - centered[:, None, :]), axis=0)
    errors = np.vstack([errors, np.mean(np.square(centered), axis=0)])
    scores = 1 - errors / y.var(axis=0)
    return np.column_stack([scores[:, :-1].mean(axis=1), scores[:, -1]]), y


def collect(subjects, visits, halves, ref, graphs):
    mapping = pilot.input_map(ref.shape[-1], key=10)
    tri = np.triu_indices(ref.shape[-1], 1)
    records = []
    for v in range(2):
        for h, half in enumerate(halves):
            for i, person in enumerate(subjects):
                ident = 10000 + len(subjects)*v + i
                source = ref[h, v, i]
                model, slopes = training_path(source, mapping, ident)
                source_scores, labels = test_path(source, mapping, ident, model, slopes)
                source_edges = source[tri]
                # Many released numerical reruns yield exactly the same off-diagonal
                # matrix. Score identical matrices once within this frozen source.
                edge_key = lambda raw: np.ascontiguousarray(raw[tri], dtype=np.float64).tobytes()
                source_key = edge_key(source)
                cache = {source_key: (source_scores, 1.)}
                replacements = [("split_half", halves[1-h], -1, ref[1-h, v, i])]
                replacements.extend(
                    ("numerical_perturbation", half, k, graphs.loc[(person, visits[v, i], half, f"sim-{k}")])
                    for k in range(20)
                )
                for rung, target_half, k, raw in replacements:
                    key = edge_key(raw)
                    if key not in cache:
                        target_scores, other_labels = test_path(raw, mapping, ident, model, slopes)
                        np.testing.assert_array_equal(labels, other_labels)
                        np.testing.assert_array_equal(source_scores[-1], target_scores[-1])
                        similarity = np.corrcoef(source_edges, raw[tri])[0, 1]
                        cache[key] = target_scores, similarity
                    target_scores, similarity = cache[key]
                    for j, alpha in enumerate(PENALTIES):
                        for t, task in enumerate(TASKS):
                            records.append(("nki", task, rung, person, person, alpha,
                                            source_scores[j, t], target_scores[j, t], v+1, v+1,
                                            half, target_half, k, int(key == source_key), similarity))
            print(f"Finished visit {v+1}, source {half}", flush=True)
    return pd.DataFrame(records, columns=[
        "cohort", "task", "rung", "target_id", "source_id", "alpha", "source_r2", "target_r2",
        "source_visit", "target_visit", "source_half", "target_half", "run",
        "identical_offdiag", "similarity",
    ])


def check(table):
    keys = ["target_id", "source_visit", "source_half"]
    assert len(table) == 25 * 2 * 2 * 21 * len(PENALTIES) * len(TASKS)
    assert not table.duplicated(keys+["rung", "run", "alpha", "task"]).any()
    assert np.isfinite(table[["source_r2", "target_r2", "similarity"]].to_numpy()).all()
    archived = pd.read_csv(ROOT / "results/numerical-repeat/runs.csv").rename(columns={
        "person": "target_id", "visit": "source_visit", "half": "source_half",
    })
    archived["rung"] = archived.condition.map({"input_mca": "numerical_perturbation", "direction_half": "split_half"})
    reference = archived.loc[archived.condition.eq("reference"), keys+["own_r2"]]
    reference = reference.rename(columns={"own_r2": "archived_source_r2"})
    transfer = archived.dropna(subset=["rung"])[keys+["rung", "run", "own_r2"]]
    sample = table.loc[table.task.eq("linear") & table.alpha.eq(10.)]
    joined = sample.merge(reference, on=keys, validate="many_to_one").merge(
        transfer, on=keys+["rung", "run"], validate="one_to_one")
    assert len(joined) == len(sample) == 2100
    max_error = float(max((joined.target_r2-joined.own_r2).abs().max(),
                          (joined.source_r2-joined.archived_source_r2).abs().max()))
    assert max_error < 1e-11
    return max_error


def main(hdf):
    subjects, visits, halves, ref, graphs = numerical.fetch_data(hdf)
    table = collect(subjects, visits, halves, ref, graphs)
    error = check(table)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "rung_pairs.csv.gz", index=False,
                 compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
    meta = dict(
        people=len(subjects), source_scans=100, split_half_pairs=100,
        perturbation_pairs=2000, rows=len(table), hdf_md5=numerical.HDF_MD5,
        identical_perturbations=int(table.query("task == 'linear' and alpha == 10 and rung == 'numerical_perturbation'").identical_offdiag.sum()),
        penalty_path="All 19 original decades 1e-14..1e4, 1e5..1e8, and an exact source-training-mean intercept.",
        test="Source and replacement receive identical four fresh role-61 signals for each participant and visit.",
        seeds="pilot.rng(11, 10000+25*zero_based_visit+subject_index, role, repeat), map key 10",
        fit_steps=pilot.FIT_N, test_steps=pilot.TEST_STREAMS*pilot.TEST_N,
        linear="Unweighted mean over ten separately scored delay predictions.",
        nonlinear="Scalar product of three input lags 1, 2, 3.",
        aggregation="Both visit and half directions for each person; 20 numerical replicates per cell.",
        checked_against="results/numerical-repeat/runs.csv, 2100 paired linear scores at penalty 10",
        max_abs_error_against_archive=error,
    )
    (OUT / "rung_metadata.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/nki/inputs.h5")
    main(parser.parse_args().source)
