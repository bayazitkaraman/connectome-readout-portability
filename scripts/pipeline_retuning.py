"""Post hoc donor-only retuning of four Beijing delayed-recall pipelines."""
import json

from pipeline_robustness import DATA, OUT, pilot, read_graphs, transfer
import numpy as np
import pandas as pd


GRID = transfer.PENALTIES
BOOTSTRAP_SEED = 20261120


def cases():
    graphs, ids, _ = read_graphs("bnu1")
    native = pilot.input_map(graphs.shape[-1], 10)
    yield "native", graphs, ids, native
    yield "log1p", np.log1p(graphs.astype(float)), ids, native
    sites = pd.read_csv(OUT / "scale_input_sites.csv")
    for scale in (200, 300):
        with np.load(DATA / f"bnu1_schaefer{scale}.npz", allow_pickle=False) as archive:
            mat = archive["weights"]
            people = archive["subjects"].astype(str)
        np.testing.assert_array_equal(people, ids)
        w = np.zeros((1, scale))
        w[0, sites[f"parcel_{scale}"].to_numpy() - 1] = sites.input_weight
        assert np.count_nonzero(w) == 8
        yield f"schaefer_{scale}", mat, people, (w, np.flatnonzero(w[0] == 0))


def score_matrix(models, test, a):
    # Evaluate every source using identical target inputs. The source index is
    # the first axis and each of the ten delays gets its own variance baseline.
    x, y = test
    w = np.stack([m.weights(a) for m in models])
    b = np.stack([m.my - m.mx @ s for m, s in zip(models, w)])
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    xx, xy = xc.T @ xc / len(x), xc.T @ yc / len(x)
    yvar = y.var(axis=0)
    delta = np.einsum("p,spk->sk", x.mean(axis=0), w) + b - y.mean(axis=0)
    wx = (xx @ w.transpose(1, 0, 2).reshape(len(xx), -1)).reshape(
        len(xx), len(models), y.shape[1]).transpose(1, 0, 2)
    mse = yvar + delta ** 2 + (w * wx).sum(axis=1) - 2 * np.einsum("spk,pk->sk", w, xy)
    scores = (1 - mse / yvar).mean(axis=1)
    direct = (1 - np.mean((models[0].predict(x, a) - y) ** 2, axis=0) / yvar).mean()
    np.testing.assert_allclose(scores[0], direct, atol=1e-10, rtol=1e-10)
    return scores


def evaluate(name, graphs, ids, mapping, archived):
    n = len(ids)
    models, losses = [], []
    for i in range(n):
        src, dest = (pilot.reservoir(graphs[v, i], .90) for v in range(2))
        train = pilot.simulate(src, mapping, 10000 + i, 0, pilot.FIT_N)
        model = pilot.RidgeFamily.from_design(train)
        x, y = pilot.simulate(dest, mapping, 10000 + n + i, 1, pilot.VALID_N)
        losses.append([np.mean((model.predict(x, a) - y) ** 2) / pilot.SIGNAL_VAR for a in GRID])
        models.append(model)
    losses = np.asarray(losses)
    scores = archived[(archived.scenario == name) & (archived.cohort == "bnu1")]
    assert len(scores) == n*n and (scores.relative_ridge == 1.).all()
    key = scores.set_index(["target", "source"])
    max_abs_error = 0.
    rows = []
    for t, person in enumerate(ids):
        target = pilot.reservoir(graphs[1, t], .90)
        test = pilot.combine([pilot.simulate(target, mapping, 10000 + n + t, 61, pilot.TEST_N, k)
                              for k in range(pilot.TEST_STREAMS)])
        selected = {}
        for d in range(n):
            if d != t:
                keep = (np.arange(n) != t) & (np.arange(n) != d)
                assert keep.sum() == n - 2
                selected[d] = GRID[losses[keep].mean(axis=0).argmin()]
        by_penalty = {a: score_matrix(models, test, a) for a in set(selected.values()) | {1.}}
        prior = key.loc[person].loc[ids]
        max_abs_error = max(max_abs_error, np.max(np.abs(prior.r2 - by_penalty[1.])))
        for d, a in selected.items():
            score = by_penalty[a]
            rows.append(dict(scenario=name, target_id=person, other_id=ids[d], relative_penalty=a,
                             own_r2=score[t], other_r2=score[d], fixed_own_r2=by_penalty[1.][t],
                             fixed_other_r2=by_penalty[1.][d], similarity=prior.iloc[d].similarity))
    assert max_abs_error < 1e-8, (name, max_abs_error)
    validation = pd.DataFrame(losses, index=ids, columns=GRID).rename_axis("person").reset_index()
    validation = validation.melt(id_vars="person", var_name="relative_penalty", value_name="validation_nmse")
    validation.insert(0, "scenario", name)
    return pd.DataFrame(rows), validation, float(max_abs_error)


def summaries(pairs):
    people = []
    for (scenario, subject), frame in pairs.groupby(["scenario", "target_id"], sort=False):
        nearest = frame.similarity == frame.similarity.max()
        row = dict(scenario=scenario, target_id=subject, nearest_id="|".join(frame.loc[nearest, "other_id"]))
        for prefix in ("", "fixed_"):
            own = frame[prefix + "own_r2"].mean()
            other = frame[prefix + "other_r2"].mean()
            best = frame.loc[nearest, prefix + "other_r2"].mean()
            row.update({prefix + k: v for k, v in dict(own=own, mean_other=other, nearest=best,
                                                       nearest_minus_mean_other=best-other,
                                                       own_minus_mean_other=own-other).items()})
        people.append(row)
    people = pd.DataFrame(people)
    metrics = ["own", "mean_other", "nearest", "nearest_minus_mean_other", "own_minus_mean_other"]
    rows = []

    def summarize(scenario, metric, mode, values, draws):
        v = np.asarray(values)
        lo, hi = np.quantile(v[draws].mean(axis=1), [.025, .975])
        rows.append(dict(scenario=scenario, metric=metric, mode=mode, n=len(v), mean=v.mean(), ci_low=lo,
                         ci_high=hi, median=np.median(v), people_positive=int((v > 0).sum())))

    for scenario, group in people.groupby("scenario", sort=False):
        group = group.sort_values("target_id")
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(group), size=(10000, len(group)))
        for m in metrics:
            for rule, values in (("retuned", group[m]), ("fixed1", group["fixed_" + m]),
                                 ("retuned_minus_fixed1", group[m] - group["fixed_" + m])):
                summarize(scenario, m, rule, values, draws)
    diffs = (people[people.scenario == "schaefer_300"].set_index("target_id")[metrics] -
             people[people.scenario == "schaefer_200"].set_index("target_id")[metrics]).sort_index()
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(diffs), size=(10000, len(diffs)))
    for m in metrics:
        summarize("schaefer_300_minus_200", m, "retuned", diffs[m], draws)
    return people, pd.DataFrame(rows)


def main():
    archive = pd.read_csv(OUT / "pair_scores.csv", dtype={"target": str, "source": str})
    frames, losses, checks = [], [], {}
    for name, graphs, ids, mapping in cases():
        print(f"Retuning Beijing {name}", flush=True)
        pairs, curve, error = evaluate(name, graphs, ids, mapping, archive)
        frames.append(pairs)
        losses.append(curve)
        checks[name] = error
    pairs = pd.concat(frames, ignore_index=True)
    people, summary = summaries(pairs)
    penalties = pairs.groupby(["scenario", "relative_penalty"]).size().rename("pairs").reset_index()
    outputs = dict(pairs=pairs, people=people, summary=summary, penalties=penalties,
                   validation=pd.concat(losses, ignore_index=True))
    for name, df in outputs.items():
        df.to_csv(OUT / f"retuned_{name}.csv", index=False)
    info = dict(status="Post hoc donor-only regularization sensitivity",
                method="Train on source visit, validate paired transfer on 55 other people; common alpha for own and foreign",
                stages="Train role 0, validation role 1, test role 61; original stream IDs and atlas-matched inputs",
                selection="Minimum mean validation MSE across other people and ten equally weighted delays",
                ridge_grid=GRID.tolist(), matched_input_file="scale_input_sites.csv",
                anatomical_selection="Maximum frozen fixed-pipeline edge correlation, independent of recall scores",
                bootstrap="10,000 target-person resamples, conditional on donor bank", bootstrap_seed=BOOTSTRAP_SEED,
                fixed1_verified_rows_per_scenario=57*57, max_abs_fixed1_discrepancy=checks)
    (OUT / "retuned_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(penalties.to_string(index=False))
    print(summary.query("mode == 'retuned'").round(5).to_string(index=False))


if __name__ == "__main__":
    main()
