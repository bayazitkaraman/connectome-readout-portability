"""Compare NKI numerical and split-half state changes with size-matched isotropic noise."""
import json

import nki_numerical_repeat as numerical
from run_visit_portability_path import both_labels, fresh_test, OUT, pilot
from state_shift_visits import evaluate_shift
import numpy as np
import pandas as pd


ALPHAS = (0.001, 10.0)
KEYS = ["cohort", "task", "rung", "source_id", "target_id", "source_visit", "target_visit",
        "source_half", "target_half", "run", "alpha", "identical_offdiag"]


def evaluate_technical():
    subjects, visits, halves, references, graphs = numerical.fetch_data(numerical.ROOT/"data/nki/inputs.h5")
    mapping = pilot.input_map(references.shape[-1], key=10)
    upper = np.triu_indices(references.shape[-1], 1)
    rows = []
    for v in range(2):
        for h, half in enumerate(halves):
            for i, person in enumerate(subjects):
                source = references[h, v, i]
                source_net = numerical.reservoir(source)
                stream_id = 10000 + len(subjects)*v + i
                x_train, y_train = pilot.simulate(source_net, mapping, stream_id, 0, pilot.FIT_N)
                model = pilot.RidgeFamily.from_design((x_train, both_labels(y_train)))
                source_states, y = fresh_test(source_net, mapping, stream_id)
                np.testing.assert_allclose(model.scale, x_train.var(axis=0).mean(), rtol=1e-12)
                source_key = source[upper].astype("float64").tobytes()
                cache = {source_key: evaluate_shift(source_states, source_states, y, model, ALPHAS)}
                replacements = [("split_half", halves[1-h], -1, references[1-h, v, i])]
                replacements.extend(
                    ("numerical_perturbation", half, k, graphs.loc[person, visits[v, i], half, f"sim-{k}"])
                    for k in range(20)
                )
                for rung, target_half, run, graph in replacements:
                    key = graph[upper].astype("float64").tobytes()
                    if key not in cache:
                        states, labels = fresh_test(numerical.reservoir(graph), mapping, stream_id)
                        np.testing.assert_array_equal(y, labels)
                        cache[key] = evaluate_shift(source_states, states, y, model, ALPHAS)
                    for result in cache[key]:
                        rows.append(dict(cohort="nki", rung=rung, source_id=person, target_id=person,
                                         source_visit=v+1, target_visit=v+1, source_half=half,
                                         target_half=target_half, run=run, identical_offdiag=int(key==source_key),
                                         **result))
    return pd.DataFrame(rows)


def verify(pairs):
    expected = pd.read_csv(OUT/"rung_pairs.csv.gz", dtype={"source_id": str, "target_id": str})
    expected = expected.loc[expected.alpha.isin(ALPHAS), KEYS+["source_r2", "target_r2"]]
    old = expected.rename(columns={"source_r2": "old_source", "target_r2": "old_target"})
    joined = pairs.merge(old, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    assert len(joined)==8400 and joined._merge.eq("both").all()
    source_error = np.abs(joined.source_r2-joined.old_source).max()
    target_error = np.abs(joined.target_r2-joined.old_target).max()
    np.testing.assert_allclose(joined.source_r2, joined.old_source, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(joined.target_r2, joined.old_target, rtol=1e-10, atol=1e-10)
    identical = pairs[pairs.identical_offdiag.eq(1)]
    assert len(identical)==986*2*len(ALPHAS) and identical.delta_rms.eq(0).all()
    np.testing.assert_array_equal(identical.source_r2, identical.isotropic_r2)
    np.testing.assert_allclose(pairs.delta_rms/pairs.train_state_rms, pairs.delta_fraction)
    return dict(source_count=100, numerical_graph_pairs=2000, split_half_graph_pairs=100,
                checked_rows=len(joined), exact_numerical_graphs=986, held_out_stream_role=61,
                max_source_r2_discrepancy=source_error, max_target_r2_discrepancy=target_error)


if __name__ == "__main__":
    pairs = evaluate_technical()
    checks = verify(pairs)
    pairs.to_csv(OUT/"shift_technical_pairs.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})
    (OUT/"shift_technical_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
