"""Independent audit of the frozen-readout feasibility output, using raw matrices and direct SVD."""
from pathlib import Path
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh, svd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
PENALTIES = 10.0 ** np.arange(-12, 0, 2)
SEED = 20261119


def stream(tag, *indices):
    return np.random.default_rng([SEED, tag, *map(int, indices)])


def fixtures(a, b, person, role, length, block=0):
    u = stream(11, person, role, block).uniform(-1, 1, length + 200)
    # Each call starts at a separate zero state; W is symmetric and x is a column here.
    x = np.zeros((len(u), len(b)))
    previous = np.zeros(len(b))
    for t, value in enumerate(u):
        x[t] = np.tanh(a @ previous + value * b)
        previous = x[t]
    # Lag k at time t uses u[t-k], and never indexes across a stream boundary.
    y = np.column_stack([u[200 - lag : len(u) - lag] for lag in range(1, 11)])
    return x[200:, b == 0], y


def sampled(data, n):
    x, y = data
    ix = np.linspace(0, len(x) - 1, n, dtype=int)
    return x[ix], y[ix]


def join(parts):
    x, y = zip(*parts)
    return np.vstack(x), np.vstack(y)


class SingularRidge:
    def __init__(self, data):
        x, y = data
        self.xbar = x.mean(axis=0)
        self.ybar = y.mean(axis=0)
        xc = x - self.xbar
        self.scale = (xc ** 2).sum() / x.shape[1]
        u, self.s, vh = svd(xc, full_matrices=False)
        self.right = vh.T
        self.projected_y = u.T @ (y - self.ybar)

    def predict(self, x, penalty):
        shrinkage = self.s / (self.s ** 2 + self.scale * penalty)
        coeff = self.right @ (shrinkage[:, None] * self.projected_y)
        return (x - self.xbar) @ coeff + self.ybar

    def tune(self, data, penalties=PENALTIES):
        x, y = data
        mses = [np.mean((self.predict(x, reg) - y) ** 2) for reg in penalties]
        return penalties[np.argmin(mses)]


def audit_expanded_grid(folds, fit, valid, networks, b):
    penalties = 10.0 ** np.arange(-14, 5, dtype=float)
    reported = pd.read_csv(R / "sensitivity_participant_scores.csv")
    reported = reported[reported.map_key == 10].set_index(["participant", "decoder", "delay"]).sort_index()
    models, scores, selected = {}, {"shared": [], "individual": []}, []
    deltas = {"shared": [], "individual": []}
    for f, (train, inner, hold) in folds.items():
        inner_model = SingularRidge(join([sampled(fit[p], 56) for p in inner]))
        alpha = inner_model.tune(join([sampled(valid[p], 84) for p in hold]), penalties)
        models[f] = SingularRidge(join([sampled(fit[p], 42) for p in train])), alpha
    own_lowest = 0
    for pid in sorted(networks):
        f = int(reported.loc[(pid, "shared"), "fold"].iloc[0])
        x, y = join([fixtures(networks[pid], b, pid, 2, 512, rep) for rep in range(4)])
        individual = SingularRidge(fit[pid])
        own_alpha = individual.tune(valid[pid], penalties)
        own_lowest += int(np.isclose(own_alpha, 1e-14, rtol=1e-10, atol=0))
        for method, (model, reg) in (("individual", (individual, own_alpha)), ("shared", models[f])):
            rows = reported.loc[(pid, method)]
            if not np.isclose(rows.relative_ridge.iloc[0], reg, rtol=1e-10, atol=0):
                selected.append((pid, method, rows.relative_ridge.iloc[0], reg))
            r2 = 1 - ((y - model.predict(x, reg)) ** 2).mean(axis=0) / y.var(axis=0)
            scores[method].append(r2)
            deltas[method].append(np.abs(r2 - rows.r2.to_numpy()))
    shared, own = np.asarray(scores["shared"]), np.asarray(scores["individual"])
    summary = pd.read_csv(R / "sensitivity_summary.csv", dtype={"delays": str})
    summary = summary[(summary.map_key == 10) & (summary.decoder == "shared")].set_index("delays")
    print("Expanded regularization range, original input map (key 10); all 70 reconstructed with direct SVD.")
    print(f"  minimum penalty selected for {own_lowest}/70 independent individual fits, "
          f"{len(selected)} penalty selection discrepancies in all shared and individual comparisons.")
    for label, slc in (("1-10", slice(None)), ("9", 8)):
        gap = float((own[:, slc]-shared[:, slc]).mean())
        print(f"  delays {label}: SVD shared R2 {shared[:,slc].mean():.9f}, individual R2 {own[:,slc].mean():.9f}, "
              f"gap {gap:.9f}. Stored gap {summary.loc[label,'r2_loss']:.9f}, change {gap-summary.loc[label,'r2_loss']:.6g}.")
    print(f"  Largest per-person/per-delay absolute R2 discrepancy: individual {np.max(deltas['individual']):.6g}, "
          f"shared {np.max(deltas['shared']):.6g}.")
    if selected:
        print("  Selected-penalty discrepancies (person, decoder, stored, SVD):", selected)


def main():
    raw = np.load(ROOT / "data/lausanne_scale1.npz")["weights"]
    frame = pd.read_csv(R / "participant_delay_scores.csv")
    summary = pd.read_csv(R / "summary.csv", dtype={"delays": str})
    run = json.loads((R / "run_info.json").read_text(encoding="utf-8"))
    assert len(frame) == 5600
    assert not frame.duplicated(["gain", "participant", "delay", "decoder"]).any()
    assert set(frame.test_rows) == {2048}
    for method, rows in frame.groupby("decoder"):
        n = 131712 if method == "shared_all_rows" else 2352
        assert set(rows.fit_rows) == {n}
    ids = set(range(1, 71))
    perm = stream(12).permutation(np.arange(1, 71))
    by_fold = {}
    for j, test in enumerate(np.array_split(perm, 5)):
        fold = j + 1
        record = run["split_membership"][j]
        assert record["fold"] == fold and set(record["test"]) == set(test)
        train = sorted(ids - set(test))
        inner = stream(13, j).permutation(train)
        assert record["training"] == train
        assert set(inner[:42]) == set(record["inner_training"])
        assert set(inner[42:]) == set(record["inner_validation"])
        assert set(frame.loc[frame.fold == fold, "participant"]) == set(test)
        by_fold[fold] = (train, inner[:42], inner[42:])
    ref = frame.query("gain == 0.9 and decoder == 'individual'").sort_values(["participant", "delay"])
    common = frame.query("gain == 0.9 and decoder == 'shared'").sort_values(["participant", "delay"])
    assert np.array_equal(ref[["participant","delay"]], common[["participant","delay"]])
    pair = pd.DataFrame({"participant": ref.participant.to_numpy(), "delay": ref.delay.to_numpy(),
                         "r2_loss": ref.r2.to_numpy()-common.r2.to_numpy(),
                         "nmse_loss": common.normalized_mse.to_numpy()-ref.normalized_mse.to_numpy()})
    draws = stream(14).choice(70, (10000,70), replace=True)
    for lag, rows in [("1-10", pair)] + [(str(k), g) for k, g in pair.groupby("delay")]:
        people = rows.groupby("participant")[["r2_loss","nmse_loss"]].mean()
        record = summary[(summary.gain == .9) & (summary.decoder == "shared") & (summary.delays == lag)].iloc[0]
        assert np.isclose(record.people_with_positive_loss, (people.r2_loss>0).sum())
        for metric in people:
            v = people[metric].to_numpy()
            expected = np.r_[v.mean(),np.percentile(v[draws].mean(axis=1),[2.5,97.5])]
            np.testing.assert_allclose(expected, [record[metric],record[metric+"_ci_low"],record[metric+"_ci_high"]], atol=2e-13,rtol=0)

    # Audit both an extreme result and a typical person from an independently trained outer fold.
    worst = common.loc[common.r2.idxmin()]
    assert worst.r2 < 0
    others = common[common.fold != worst.fold].groupby("participant").r2.mean()
    typical_id = (others - others.median()).abs().idxmin()
    people = [int(worst.participant),int(typical_id)]
    nodes=stream(10).choice(raw.shape[1],8,replace=False)
    b=np.zeros(raw.shape[1]);b[nodes[:4]]=0.5;b[nodes[4:]]=-0.5
    np.testing.assert_array_equal(np.flatnonzero(b==0)+1,run["readout_nodes"])
    assert {str(i+1): b[i] for i in nodes} == run["input_weights"]
    networks, fit, valid = {}, {}, {}
    for pid, adj in enumerate(raw, 1):
        a = adj.copy()
        np.fill_diagonal(a, 0)
        radius = max(abs(eigh(a,eigvals_only=True)))
        networks[pid] = a * (.9/radius)
        fit[pid]=fixtures(networks[pid], b, pid, 0, 2352)
        valid[pid]=fixtures(networks[pid], b, pid, 1, 1176)
    differences=[]
    models = {}
    for fold, (train, inner_fit, inner_val) in by_fold.items():
        sx = SingularRidge(join([sampled(fit[p],56) for p in inner_fit]))
        shared_alpha = sx.tune(join([sampled(valid[p],84) for p in inner_val]))
        pooled = SingularRidge(join([sampled(fit[p],42) for p in train]))
        models[fold] = pooled, shared_alpha
    recomputed = {"shared": [], "individual": []}
    tuning_differences = []
    for pid in sorted(ids):
        fold = int(common.loc[common.participant == pid,"fold"].iloc[0])
        pooled, shared_alpha = models[fold]
        individual = SingularRidge(fit[pid]); own_alpha = individual.tune(valid[pid])
        x,y = join([fixtures(networks[pid],b,pid,2,512,s) for s in range(4)])
        for method,model,penalty in (("individual",individual,own_alpha),("shared",pooled,shared_alpha)):
            pred = model.predict(x,penalty)
            err = np.mean((y-pred)**2,axis=0)
            r2=1-err/np.var(y,axis=0)
            rows=frame[(frame.gain==.9)&(frame.participant==pid)&(frame.decoder==method)].sort_values("delay")
            if not np.allclose(rows.relative_ridge,penalty,rtol=1e-12,atol=0):
                tuning_differences.append((pid,method,float(rows.relative_ridge.iloc[0]),float(penalty)))
            diff=max(np.max(np.abs(r2-rows.r2)),np.max(np.abs(3*err-rows.normalized_mse)))
            differences.append(float(diff))
            recomputed[method].append(r2)
            if pid in people:
                print(f"Participant {pid:02}, fold {fold}, {method:10}, relative ridge {penalty:g}, "
                      f"lag-9 R2 {r2[8]:.9f}, minimum R2 {min(r2):.9f}, max score discrepancy {diff:.3g}")
    print(f"Verified five outer and inner membership sets, 5600 unique output rows and all 11 primary paired summaries and bootstrap intervals.")
    print(f"All 70 holdouts direct recurrence + centered design SVD reconstruction: maximum absolute score discrepancy {max(differences):.6g}.")
    print("Tuning differences (person, model, official, SVD):",tuning_differences)
    own_r2=np.asarray(recomputed["individual"]); shared_r2=np.asarray(recomputed["shared"])
    print(f"Independent full-cohort R2: shared {shared_r2.mean():.9f}, individual {own_r2.mean():.9f}, paired difference {(own_r2-shared_r2).mean():.9f}.")
    print(f"Independent full-cohort delay-9 R2: shared {shared_r2[:,8].mean():.9f}, individual {own_r2[:,8].mean():.9f}.")
    print(f"Worst primary shared row: participant {int(worst.participant)}, lag {int(worst.delay)}, R2 {float(worst.r2):.9f}.")
    # Pooled moment fits and independently reconstructed trajectories should agree
    # with the direct design-matrix calculation to better than reporting precision.
    assert not tuning_differences
    assert max(differences) < 1e-7
    primary = summary[(summary.gain == .9) & (summary.decoder == "shared") & (summary.delays == "1-10")].iloc[0]
    assert abs((own_r2-shared_r2).mean() - primary.r2_loss) < 1e-8
    audit_expanded_grid(by_fold, fit, valid, networks, b)


if __name__ == "__main__":
    main()

