"""Post hoc upper-ridge-boundary and source-local transfer diagnostics."""
import json
from functools import lru_cache

from nonlinear_recall import (BOOTSTRAP_SEED, OUT, TARGET_VARIANCE, pilot,
                              product_data, read_graphs, testing_data, transfer)
import numpy as np
import pandas as pd


GRID = np.union1d(transfer.PENALTIES, np.r_[10. ** np.arange(0, 8.01, .25), np.inf])


def prediction(model, x, alpha):
    if np.isinf(alpha):
        return np.broadcast_to(model.my, (len(x), 1))
    return model.predict(x, alpha)


def r2(model, test, alpha):
    x, y = test
    return float(1 - np.mean((prediction(model, x, alpha) - y) ** 2) / np.var(y))


def calculate(cohort, primary):
    graphs, ids, _ = read_graphs(cohort)
    n = len(ids)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    models, targets, selves, losses, local, curves = [], [], [], [], [], []
    for t, person in enumerate(ids):
        src = pilot.reservoir(graphs[0, t], transfer.GAIN)
        tar = pilot.reservoir(graphs[1, t], transfer.GAIN)
        source_id, target_id = 10000 + t, 10000 + n + t
        fit = product_data(src, mapping, source_id, 0, pilot.FIT_N)
        valid_source = product_data(src, mapping, source_id, 1, pilot.VALID_N)
        valid_target = product_data(tar, mapping, target_id, 1, pilot.VALID_N)
        target_test = testing_data(tar, mapping, target_id)
        source_test = testing_data(src, mapping, target_id)
        np.testing.assert_array_equal(source_test[1], target_test[1])
        model = pilot.RidgeFamily.from_design(fit)
        local_alpha = model.tune(valid_source, transfer.PENALTIES)
        # Verify the extreme local-penalty transfer scores with direct augmented
        # least squares, including the intercept outside the penalty.
        x, y = fit
        xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
        ridge = np.sqrt(local_alpha * (xc ** 2).sum() / x.shape[1])
        augmented = np.vstack([xc, ridge * np.eye(x.shape[1])])
        rhs = np.vstack([yc, np.zeros((x.shape[1], 1))])
        independent_w = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        relative_differences = []
        local_predictions = []
        for test in (source_test, target_test):
            predicted = model.predict(test[0], local_alpha)
            other_fit = (test[0] - x.mean(axis=0)) @ independent_w + y.mean(axis=0)
            discrepancy = np.linalg.norm(other_fit - predicted) / np.linalg.norm(predicted)
            assert discrepancy < 1e-6, (cohort, person, discrepancy)
            relative_differences.append(float(discrepancy))
            local_predictions.append(float(np.std(predicted) / np.std(test[1])))
        local.append(dict(source_local_alpha=local_alpha, source_local_r2=r2(model, source_test, local_alpha),
                          source_local_on_target_r2=r2(model, target_test, local_alpha),
                          local_prediction_sd_on_source=local_predictions[0],
                          local_prediction_sd_on_target=local_predictions[1],
                          local_reference_max_relative_discrepancy=max(relative_differences)))
        validations = [np.mean((prediction(model, valid_target[0], a) - valid_target[1]) ** 2)
                       / TARGET_VARIANCE for a in GRID]
        losses.append(validations)
        for a, v in zip(GRID, validations):
            curves.append(dict(cohort=cohort, target_id=person, alpha=a, validation_nmse=v))
        models.append(model)
        targets.append(target_test)
        selves.append(source_test)
    losses = np.asarray(losses)
    primary = primary[primary.cohort == cohort].set_index(["target_id", "other_id"])
    locked = np.isin(GRID, transfer.PENALTIES)
    assert locked.sum() == len(transfer.PENALTIES)
    people, pairs = [], []
    found, total = 0, 0
    for t, person in enumerate(ids):
        @lru_cache(None)
        def own(alpha):
            return r2(models[t], selves[t], alpha), r2(models[t], targets[t], alpha)

        candidates = []
        for d, donor in enumerate(ids):
            if t == d:
                continue
            eligible = (np.arange(n) != t) & (np.arange(n) != d)
            v = losses[eligible].mean(axis=0)
            a = GRID[v.argmin()]
            locked_alpha = GRID[locked][v[locked].argmin()]
            previous = primary.loc[(person, donor)]
            assert locked_alpha == previous.alpha
            rep_source, rep_own = own(locked_alpha)
            rep_foreign = r2(models[d], targets[t], locked_alpha)
            np.testing.assert_allclose([rep_source, rep_own, rep_foreign],
                                       [previous.source_self_r2, previous.own_r2, previous.other_r2],
                                       rtol=1e-9, atol=1e-12)
            self_r2, own_r2 = own(a)
            foreign = r2(models[d], targets[t], a)
            found += int(np.isinf(a))
            total += 1
            candidates.append(dict(cohort=cohort, target_id=person, other_id=donor,
                                   dense_alpha=a, locked_alpha=locked_alpha,
                                   source_self_r2=self_r2, own_r2=own_r2, other_r2=foreign,
                                   nearest_foreign=previous.nearest_foreign,
                                   donor_validation_nmse=v.min(), exact_mean_validation_nmse=v[-1]))
        block = pd.DataFrame(candidates)
        nearest = block[block.nearest_foreign].other_r2.mean()
        mean_other = block.other_r2.mean()
        people.append(dict(cohort=cohort, target_id=person, source_self_r2=block.source_self_r2.mean(),
                           own_r2=block.own_r2.mean(), nearest_r2=nearest, mean_other_r2=mean_other,
                           nearest_minus_mean_other=nearest - mean_other, **local[t],
                           local_source_r2_minus_target=local[t]["source_local_r2"] -
                           local[t]["source_local_on_target_r2"]))
        pairs.append(block)
        own.cache_clear()
    print(f"{cohort}: {found} of {total} cells chose the exact source-mean limit", flush=True)
    return pd.concat(pairs, ignore_index=True), pd.DataFrame(people), pd.DataFrame(curves)


def main():
    primary = pd.read_csv(OUT / "pair_scores.csv", dtype={"target_id": str, "other_id": str})
    blocks = [calculate(c, primary) for c in ("nki", "bnu1")]
    pairs, people, curves = [pd.concat(frames, ignore_index=True) for frames in zip(*blocks)]
    summary = []
    for cohort, group in people.groupby("cohort", sort=False):
        n = len(group)
        ix = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(10000, n))
        for col in ["source_self_r2", "own_r2", "nearest_r2", "mean_other_r2", "nearest_minus_mean_other",
                    "source_local_r2", "source_local_on_target_r2", "local_source_r2_minus_target"]:
            v = group[col].to_numpy()
            lo, hi = np.percentile(v[ix].mean(axis=1), [2.5, 97.5])
            summary.append(dict(cohort=cohort, metric=col, n=n, mean=v.mean(), ci_low=lo, ci_high=hi,
                                median=np.median(v), minimum=v.min(), maximum=v.max(),
                                people_positive=int((v > 0).sum())))
    summary = pd.DataFrame(summary)
    for name, frame in [("pairs", pairs), ("people", people), ("summary", summary),
                        ("validation", curves)]:
        frame.to_csv(OUT / f"boundary_{name}.csv", index=False)
    meta = dict(status="Post hoc diagnostic after observing the original high-penalty selections",
                finite_alpha=GRID[np.isfinite(GRID)].tolist(),
                additional_candidate="Infinity uses the exact source training mean with zero coefficients",
                selection="Lowest validation error on own-to-repeat transfer among all people except target and donor",
                local_selection="Original 19 penalties; source-local validation; same frozen fit tested on both graphs",
                check="All primary pair penalties and original-grid source/own/foreign scores replayed",
                independent_local_solve="Direct augmented least squares, intercept unpenalized; relative prediction difference bounded by 1e-6 on both graphs",
                local_prediction_sd_units="Standard deviations of held-out target values on the same streams")
    (OUT / "boundary_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
