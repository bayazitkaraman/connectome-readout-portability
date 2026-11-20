"""Match frozen source and destination test scores on each point of the ridge path."""
import json
from dataclasses import dataclass

from nonlinear_recall import anatomical_similarity, pilot, read_graphs, transfer
import numpy as np
import pandas as pd


OUT = transfer.ROOT / "results" / "portability-path"
ALPHA = np.append(np.logspace(-14, 8, 23), np.inf)
TASKS = ("linear", "nonlinear")


def both_labels(delayed):
    return np.column_stack([delayed, delayed[:, :3].prod(axis=1)])


def fresh_test(net, mapping, stream_id):
    x, y = pilot.combine([pilot.simulate(net, mapping, stream_id, 61, pilot.TEST_N, k)
                          for k in range(pilot.TEST_STREAMS)])
    return x, both_labels(y)


@dataclass
class TestDesign:
    rx: np.ndarray
    ry: np.ndarray
    xbar: np.ndarray
    ybar: np.ndarray
    variance: np.ndarray
    rows: int

    @classmethod
    def from_data(cls, x, y):
        # QR preserves residual norms, including the near-cancelling terms in
        # source scores at the weakest penalties. Squared Gram matrices do not.
        xbar, ybar = x.mean(axis=0), y.mean(axis=0)
        compact = np.linalg.qr(np.column_stack([x - xbar, y - ybar]), mode="r")
        return cls(compact[:, :x.shape[1]], compact[:, x.shape[1]:], xbar, ybar,
                   y.var(axis=0), len(y))

    def score(self, model, weights):
        features, penalties, outputs = weights.shape
        predicted = (self.rx @ weights.reshape(features, -1)).reshape(-1, penalties, outputs)
        residual = predicted - self.ry[:, None, :]
        bias = ((self.xbar - model.mx) @ weights.reshape(features, -1)).reshape(penalties, outputs)
        bias += model.my - self.ybar
        per_output = 1 - (np.sum(residual**2, axis=0) / self.rows + bias**2) / self.variance
        per_output[-1] = -((model.my - self.ybar)**2) / self.variance
        return np.column_stack([per_output[:, :-1].mean(axis=1), per_output[:, -1]])


def fit_source(net, mapping, stream_id):
    x, delayed = pilot.simulate(net, mapping, stream_id, 0, pilot.FIT_N)
    model = pilot.RidgeFamily.from_design((x, both_labels(delayed)))
    weights = [model.weights(a) for a in ALPHA[:-1]] + [np.zeros_like(model.proj)]
    return model, np.stack(weights, axis=1)


def direct_check(model, x, y, result):
    for k, alpha in enumerate(ALPHA):
        prediction = model.my if np.isinf(alpha) else model.predict(x, alpha)
        direct = 1 - ((prediction - y)**2).mean(axis=0) / y.var(axis=0)
        expected = [direct[:-1].mean(), direct[-1]]
        np.testing.assert_allclose(result[k], expected, rtol=2e-7, atol=5e-8)


def evaluate(cohort):
    graphs, ids, half = read_graphs(cohort)
    n = len(ids)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    similarities = anatomical_similarity(graphs)
    foreign_sim = similarities.copy()
    np.fill_diagonal(foreign_sim, -np.inf)
    nearest = foreign_sim == foreign_sim.max(axis=1, keepdims=True)
    target_designs, target_labels = [], []
    for t, graph in enumerate(graphs[1]):
        net = pilot.reservoir(graph, transfer.GAIN)
        x, y = fresh_test(net, mapping, 10000 + n + t)
        target_designs.append(TestDesign.from_data(x, y))
        target_labels.append(y)

    blocks = []
    for d, graph in enumerate(graphs[0]):
        net = pilot.reservoir(graph, transfer.GAIN)
        model, weights = fit_source(net, mapping, 10000 + d)
        for t, design in enumerate(target_designs):
            stream = 10000 + n + t
            x, y = fresh_test(net, mapping, stream)
            np.testing.assert_array_equal(y, target_labels[t])
            source_scores = TestDesign.from_data(x, y).score(model, weights)
            target_scores = design.score(model, weights)
            np.testing.assert_array_equal(source_scores[-1], target_scores[-1])
            if (d, t) in {(0, 0), (n-1, n-1), (0, n-1)}:
                direct_check(model, x, y, source_scores)
                xt, yt = fresh_test(pilot.reservoir(graphs[1, t], transfer.GAIN), mapping, stream)
                direct_check(model, xt, yt, target_scores)
            for j, task in enumerate(TASKS):
                blocks.append(pd.DataFrame(dict(
                    cohort=cohort, task=task, rung="between_visit" if t == d else "stranger",
                    target_id=ids[t], source_id=ids[d], alpha=ALPHA, source_r2=source_scores[:, j],
                    target_r2=target_scores[:, j], source_visit=1, target_visit=2,
                    source_half=half, target_half=half, similarity=similarities[t, d],
                    nearest_foreign=nearest[t, d], source_train_stream=10000+d, test_stream=stream)))
        if d % 10 == 0 or d == n-1:
            print(f"{cohort} finished source {d+1}/{n}", flush=True)
    frame = pd.concat(blocks, ignore_index=True)
    assert len(frame) == n*n*len(ALPHA)*len(TASKS)
    assert np.isfinite(frame[["source_r2", "target_r2"]].to_numpy()).all()
    return frame


def archive_checks(frame):
    folder = transfer.ROOT / "results"
    checks = {}
    ids = {"target_id": str, "other_id": str}
    linear = pd.read_csv(folder/"density-controls/frozen/single_donor_pairs.csv", dtype=ids)
    linear = linear[linear.source_input.eq("original_distinct_inputs")].rename(columns={"relative_penalty": "alpha"})
    nonlinear = pd.read_csv(folder/"nonlinear-recall/pair_scores.csv", dtype=ids)
    for task, archive in zip(TASKS, (linear, nonlinear)):
        data = frame[frame.task.eq(task)]
        source_keys = ["cohort", "target_id", "source_id", "alpha"]
        own = archive.assign(source_id=archive.target_id).merge(data, on=source_keys, validate="many_to_one")
        other = archive.assign(source_id=archive.other_id).merge(data, on=source_keys, validate="one_to_one")
        assert len(own) == len(other) == len(archive) == 3792
        errors = [abs(own.own_r2 - own.target_r2).max(), abs(other.other_r2 - other.target_r2).max()]
        if task == "nonlinear":
            errors.append(abs(own.source_self_r2 - own.source_r2).max())
        assert max(errors) < 1e-10
        checks[task] = dict(original_stranger_pairs=len(other), max_absolute_difference=float(max(errors)))
    return checks


def main():
    np.testing.assert_array_equal(ALPHA[:len(transfer.PENALTIES)], transfer.PENALTIES)
    frame = pd.concat([evaluate(cohort) for cohort in ("nki", "bnu1")], ignore_index=True)
    checks = archive_checks(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0}
    frame.to_csv(OUT/"visit_pairs.csv.gz", index=False, compression=compression)
    keys = ["cohort", "task", "rung", "target_id", "alpha"]
    people = frame.groupby(keys, as_index=False)[["source_r2", "target_r2"]].mean()
    people.to_csv(OUT/"visit_people.csv.gz", index=False, compression=compression)
    (OUT/"visit_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
