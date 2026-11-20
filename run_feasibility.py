"""Compare a single cross-person readout against independently calibrated readouts."""
from pathlib import Path
import os

for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import csv
import json
import platform
import sys
from dataclasses import dataclass
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from conn2res.reservoir import EchoStateNetwork

SEED = 20261119
GAINS = (0.90, 0.70, 1.00)
DELAYS = np.arange(1, 11)
PENALTIES = np.power(10., np.arange(-12, 0, 2))
FIT_N, VALID_N, TEST_N, WARMUP = 2352, 1176, 512, 200
TEST_STREAMS = 4
SIGNAL_VAR = 1 / 3
OUT = ROOT / "results"


def rng(*keys):
    return np.random.default_rng(np.random.SeedSequence([SEED, *map(int, keys)]))


def input_map(n, key=10):
    indices = rng(key).choice(n, size=8, replace=False)
    weights = np.zeros((1, n))
    weights[0, indices] = [0.5] * 4 + [-0.5] * 4
    return weights, np.flatnonzero(weights[0] == 0)


def reservoir(raw, gain):
    largest = np.max(np.abs(np.linalg.eigvalsh(raw)))
    if largest <= 0:
        raise ValueError("Set of interregional connections is empty")
    return EchoStateNetwork(w=np.ascontiguousarray(raw * (gain / largest)),
                            activation_function="tanh", leak_rate=None)


def simulate(net, mapping, person, role, length, repeat=0):
    signal = rng(11, person, role, repeat).uniform(-1, 1, length + WARMUP)
    inputs, read_nodes = mapping
    states = net.simulate(ext_input=signal[:, None], w_in=inputs,
                          output_nodes=read_nodes)
    times = np.arange(WARMUP, len(signal))
    earlier_inputs = signal[times[:, None] - DELAYS[None, :]]
    return np.ascontiguousarray(states[WARMUP:]), earlier_inputs


def sampled(data, count):
    x, y = data
    if count == len(x):
        return x, y
    idx = np.linspace(0, len(x) - 1, count, dtype=int)
    return x[idx], y[idx]


def combine(parts):
    xs, ys = zip(*parts)
    return np.concatenate(xs), np.concatenate(ys)


@dataclass
class Moment:
    """Uncentered training moments that can be pooled exactly across people."""
    count: int
    sx: np.ndarray
    sy: np.ndarray
    xx: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_data(cls, data):
        x, y = data
        return cls(len(x), x.sum(axis=0), y.sum(axis=0), x.T @ x, x.T @ y)


def merge(stats):
    return Moment(sum(s.count for s in stats),
                  sum(s.sx for s in stats), sum(s.sy for s in stats),
                  sum(s.xx for s in stats), sum(s.xy for s in stats))


class RidgeFamily:
    def __init__(self, m):
        self.mx = m.sx / m.count
        self.my = m.sy / m.count
        xx = m.xx / m.count - np.outer(self.mx, self.mx)
        xy = m.xy / m.count - np.outer(self.mx, self.my)
        val, self.vec = np.linalg.eigh(xx)
        self.val = np.maximum(val, 0)
        self.scale = np.trace(xx) / len(xx)
        self.proj = self.vec.T @ xy

    def weights(self, relative_penalty):
        diagonal = self.val + self.scale * relative_penalty
        return self.vec @ (self.proj / diagonal[:, None])

    def predict(self, x, penalty):
        return (x - self.mx) @ self.weights(penalty) + self.my

    def tune(self, validation, penalties=PENALTIES):
        x, y = validation
        errors = [np.mean((self.predict(x, p) - y) ** 2) for p in penalties]
        return float(penalties[np.argmin(errors)])

    @classmethod
    def from_design(cls, data):
        """Use singular values directly for the weakly penalized individual fits."""
        x, y = data
        model = cls.__new__(cls)
        model.mx, model.my = x.mean(axis=0), y.mean(axis=0)
        left, singular, right_t = np.linalg.svd(x - model.mx, full_matrices=False)
        model.vec = right_t.T
        model.val = singular ** 2 / len(x)
        model.scale = model.val.sum() / x.shape[1]
        model.proj = singular[:, None] * (left.T @ (y - model.my)) / len(x)
        return model


def write_csv(filename, rows):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def folds(n):
    perm = rng(12).permutation(n)
    for fold, test in enumerate(np.array_split(perm, 5)):
        train = np.setdiff1d(np.arange(n), test)
        inner = rng(13, fold).permutation(train)
        assert len(train) == 56 and len(test) == 14 and not set(train) & set(test)
        yield fold + 1, train, inner[:42], inner[42:], test


def score(rows, gain, fold, pid, method, model, penalty, test, n_fit):
    x, y = test
    err = ((model.predict(x, penalty) - y) ** 2).mean(axis=0)
    variance = y.var(axis=0)
    for k, mse, var in zip(DELAYS, err, variance):
        rows.append(dict(gain=gain, fold=fold, participant=pid + 1,
                         delay=int(k), decoder=method, fit_rows=n_fit,
                         relative_ridge=penalty, test_rows=len(y),
                         normalized_mse=float(mse / SIGNAL_VAR),
                         r2=float(1 - mse / var)))


def evaluate(raw, gain, rows, split_log, map_key=10, penalties=PENALTIES):
    mapping = input_map(raw.shape[1], map_key)
    calibration, validation, testing = [], [], []
    for i, adj in enumerate(raw):
        net = reservoir(adj, gain)
        calibration.append(simulate(net, mapping, i + 1, 0, FIT_N))
        validation.append(simulate(net, mapping, i + 1, 1, VALID_N))
        testing.append(combine([simulate(net, mapping, i + 1, 2, TEST_N, s)
                                for s in range(TEST_STREAMS)]))

    counts = (42, 56, FIT_N)
    moments = {n: [Moment.from_data(sampled(d, n)) for d in calibration] for n in counts}
    for fold, train, inner_train, inner_valid, test in folds(len(raw)):
        if gain == GAINS[0]:
            split_log.append({"fold": fold, **{
                name: sorted((members + 1).tolist()) for name, members in
                [("training", train), ("inner_training", inner_train),
                 ("inner_validation", inner_valid), ("test", test)]}})
        inner_y = combine([sampled(validation[k], 84) for k in inner_valid])
        to_score = {}
        def pooled(inner_count, final_count):
            inner = RidgeFamily(merge([moments[inner_count][k] for k in inner_train]))
            alpha = inner.tune(inner_y, penalties)
            model = RidgeFamily(merge([moments[final_count][k] for k in train]))
            return model, alpha, len(train) * final_count

        to_score["shared"] = pooled(56, 42)
        if gain == GAINS[0]:
            to_score["shared_all_rows"] = pooled(FIT_N, FIT_N)
            group = reservoir(raw[train].mean(axis=0), gain)
            # Template tuning only sees inner-training anatomy and inner-validation people.
            inner_group = reservoir(raw[inner_train].mean(axis=0), gain)
            gid = 1000 + fold
            inner_m = Moment.from_data(simulate(inner_group, mapping, gid, 3, FIT_N))
            group_penalty = RidgeFamily(inner_m).tune(inner_y, penalties)
            group_m = Moment.from_data(simulate(group, mapping, gid, 4, FIT_N))
            to_score["group_average"] = RidgeFamily(group_m), group_penalty, FIT_N

        for i in test:
            own = RidgeFamily.from_design(calibration[i])
            alpha = own.tune(validation[i], penalties)
            score(rows, gain, fold, i, "individual", own, alpha, testing[i], FIT_N)
            for method, (model, penalty, n_fit) in to_score.items():
                score(rows, gain, fold, i, method, model, penalty, testing[i], n_fit)
    print(f"Scored all {len(raw)} held-out participants at gain {gain}", flush=True)


def summarize(rows):
    import pandas as pd
    frame = pd.DataFrame(rows)
    indexed = frame.set_index(["gain", "participant", "delay", "decoder"])
    ref = indexed.xs("individual", level="decoder")
    boot = rng(14).integers(0, len(ref.index.unique("participant")), size=(10000, 70))
    out = []
    for method in ("shared", "group_average", "shared_all_rows"):
        pred = indexed.xs(method, level="decoder")
        comp = pred[["normalized_mse","r2"]].join(
            ref[["normalized_mse","r2"]], how="left", rsuffix="_own")
        comp["nmse_loss"] = comp.normalized_mse - comp.normalized_mse_own
        comp["r2_loss"] = comp.r2_own - comp.r2
        for gain in sorted(comp.index.unique("gain")):
            table = comp.xs(gain, level="gain")
            for k in [0, *DELAYS.tolist()]:
                people = table.groupby(level="participant").mean() if k == 0 else table.xs(k, level="delay")
                people = people.sort_index()
                assert len(people) == 70
                row = dict(gain=gain, decoder=method, delays=("1-10" if k == 0 else str(k)),
                           n_people=len(people), mean_r2=people.r2.mean(),
                           mean_own_r2=people.r2_own.mean(), mean_normalized_mse=people.normalized_mse.mean(),
                           mean_own_normalized_mse=people.normalized_mse_own.mean())
                for col in ("r2_loss","nmse_loss"):
                    values = people[col].to_numpy()
                    low, high = np.quantile(values[boot].mean(axis=1), [.025,.975])
                    row.update({col: values.mean(), col+"_ci_low": low, col+"_ci_high": high})
                row["people_with_positive_loss"] = int((people.r2_loss > 0).sum())
                out.append(row)
    return out


def participant_table(rows):
    import pandas as pd
    data = pd.DataFrame(rows)
    primary = data.loc[(data.gain == GAINS[0]) & data.decoder.isin(["shared", "individual"])]
    paired = primary.pivot(index=["participant", "fold", "delay"], columns="decoder",
                           values=["r2", "normalized_mse"])
    paired.columns = ["_".join(key) for key in paired.columns]
    paired["r2_loss"] = paired.r2_individual - paired.r2_shared
    paired["normalized_mse_loss"] = paired.normalized_mse_shared - paired.normalized_mse_individual
    means = paired.groupby(level=["participant", "fold"]).mean().add_suffix("_mean_lags_1_10")
    nine = paired.xs(9, level="delay").add_suffix("_lag_9")
    return means.join(nine).reset_index().to_dict("records")


def load_adjacencies():
    with np.load(ROOT / "data" / "lausanne_scale1.npz") as f:
        raw = f["weights"].astype(np.float64, copy=True)
        assert np.array_equal(f["participant_index"], np.arange(1, 71))
    assert raw.shape == (70, 68, 68) and np.allclose(raw, raw.transpose(0,2,1))
    assert np.isfinite(raw).all() and raw.min() >= 0
    diagonal = np.arange(raw.shape[1])
    raw[:, diagonal, diagonal] = 0
    return raw


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run the expanded-penalty and input-map sensitivity checks.")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    raw = load_adjacencies()
    if args.diagnostics:
        penalties = np.logspace(-14, 4, 19)
        raw_results, summaries = [], []
        for key in [10, 101, 102]:
            scores = []
            evaluate(raw, 0.90, scores, [], map_key=key, penalties=penalties)
            raw_results.extend(dict(map_key=key, **row) for row in scores)
            summaries.extend(dict(map_key=key, **row) for row in summarize(scores))
        write_csv("sensitivity_participant_scores.csv", raw_results)
        write_csv("sensitivity_summary.csv", summaries)
        return
    rows, split_log = [], []
    for gain in GAINS:
        evaluate(raw, gain, rows, split_log)
    write_csv("participant_delay_scores.csv", rows)
    write_csv("participant_transfer.csv", participant_table(rows))
    write_csv("summary.csv", summarize(rows))
    inputs, output_indices = input_map(raw.shape[1])
    info = {"seed": SEED, "python":platform.python_version(), "numpy":np.__version__,
            "node_indices_one_based":True, "input_weights": {
                str(int(i+1)): float(inputs[0,i]) for i in np.flatnonzero(inputs[0])},
            "readout_nodes":(output_indices+1).tolist(), "split_membership": split_log}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

