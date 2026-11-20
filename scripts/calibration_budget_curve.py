"""Equal target-state budgets for parcel alignment with and without recall labels."""
import json
from time import perf_counter

from unlabeled_parcel_alignment import (BOOTSTRAP_SEED, CALIBRATION_N, CALIBRATION_ROLE,
                                        PENALTIES, fresh_test, moment_map, pilot,
                                        read_graphs, r_squared, transfer)
import numpy as np
import pandas as pd


OUT = transfer.ROOT / "results" / "calibration-budget-curve"
BUDGETS = (8, 16, 32, 64, 128, 256, 512, 1024, CALIBRATION_N)
ALPHAS = np.r_[10. ** np.arange(-14, 5), np.inf]


def aligned_predictions(source_training, calibration, feature_sets, weights, model):
    mean_source, mean_target, ratio, _, _ = moment_map(source_training, calibration)
    # Transformed features are affine, so compose their coefficients once.
    slopes = ratio[:, None] * weights
    intercept = (mean_source - model.mx) @ weights + model.my
    return [(x - mean_target) @ slopes + intercept for x in feature_sets]


def compact_mse(x, residual, correction, slopes):
    """Per-output prediction MSE for the full residual ridge path including no correction."""
    outputs = residual.shape[1]
    gram_factor = np.linalg.qr(
        np.column_stack((x - correction.mx, residual - correction.my)), mode="r")
    p = x.shape[1]
    pred = (gram_factor[:, :p] @ slopes.reshape(p, -1)).reshape(-1, len(ALPHAS) - 1, outputs)
    errors = pred - gram_factor[:, None, p:]
    finite = np.sum(errors * errors, axis=0) / len(x)
    return np.vstack((finite, np.mean(residual**2, axis=0)))


def independent_predictions(x_train, residual, x_eval, relative_alpha):
    """Fit with a least-squares augmented design, independently of the SVD ridge family."""
    n, p = x_train.shape
    x_centered = x_train - x_train.mean(axis=0)
    absolute = relative_alpha * np.mean(x_train.var(axis=0))
    augmented_x = np.vstack((x_centered, np.eye(p) * np.sqrt(n * absolute)))
    augmented_y = np.vstack((residual - residual.mean(axis=0), np.zeros((p, residual.shape[1]))))
    slopes, _, _, _ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
    return (x_eval - x_train.mean(axis=0)) @ slopes + residual.mean(axis=0)


def make_paths(cohort):
    graphs, people, _ = read_graphs(cohort)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    n_people = len(people)
    shape = (n_people, len(BUDGETS), len(ALPHAS))
    validation = np.zeros(shape)
    test = np.zeros(shape)
    frozen_scores = np.zeros((n_people, 2))
    direct_errors = []
    for i, person in enumerate(people):
        source_id, target_id = 10000 + i, 10000 + n_people + i
        source_net, target_net = [pilot.reservoir(visit[i], transfer.GAIN) for visit in graphs]
        xs_train, ys_train = pilot.simulate(source_net, mapping, source_id, 0, pilot.FIT_N)
        model = pilot.RidgeFamily.from_design((xs_train, ys_train))
        weights = model.weights(PENALTIES[cohort])
        xc, yc = pilot.simulate(target_net, mapping, target_id, CALIBRATION_ROLE, CALIBRATION_N)
        xv, yv = pilot.simulate(target_net, mapping, target_id, 1, pilot.VALID_N)
        xs, delays_s = fresh_test(source_net, mapping, target_id)
        xt, delays_t = fresh_test(target_net, mapping, target_id)
        np.testing.assert_array_equal(delays_s, delays_t)
        yt = delays_t[:, :len(pilot.DELAYS)]
        variance = yt.var(axis=0)
        frozen_scores[i] = [r_squared(yt, (x - model.mx) @ weights + model.my) for x in (xs, xt)]
        for j, budget in enumerate(BUDGETS):
            prefix_x, prefix_y = xc[:budget], yc[:budget]
            cal_prediction, val_prediction, test_prediction = aligned_predictions(
                xs_train, prefix_x, (prefix_x, xv, xt), weights, model)
            residual = prefix_y - cal_prediction
            correction = pilot.RidgeFamily.from_design((prefix_x, residual))
            slopes = np.stack([correction.weights(alpha) for alpha in ALPHAS[:-1]], axis=1)
            vmse = compact_mse(xv, yv - val_prediction, correction, slopes)
            tmse = compact_mse(xt, yt - test_prediction, correction, slopes)
            validation[i, j] = vmse.mean(axis=1)
            test[i, j] = (1 - tmse / variance).mean(axis=1)
            np.testing.assert_allclose(test[i, j, -1], r_squared(yt, test_prediction),
                                       atol=1e-13, rtol=0)
            if i in (0, len(people)-1) and budget in (8, 64, 2352):
                for alpha in (1e-8, .01):
                    k = list(ALPHAS).index(alpha)
                    val_correction = independent_predictions(prefix_x, residual, xv, alpha)
                    test_correction = independent_predictions(prefix_x, residual, xt, alpha)
                    direct_val = np.square(yv - val_prediction - val_correction).mean()
                    direct_test = r_squared(yt, test_prediction + test_correction)
                    for compact, direct in ((validation[i,j,k],direct_val),(test[i,j,k],direct_test)):
                        np.testing.assert_allclose(compact, direct, rtol=1e-8, atol=1e-9)
                        direct_errors.append(float(abs(compact-direct)))
        if not np.isfinite(validation[i]).all() or not np.isfinite(test[i]).all():
            raise ValueError(f"Nonfinite curve for {cohort} participant {person}")
    return np.asarray(people).astype(str), frozen_scores, validation, test, direct_errors


def select_paths(cohort):
    people, frozen, validation, test, errors = make_paths(cohort)
    rows, choices, losses = [], [], []
    for i, person in enumerate(people):
        source_r2, frozen_r2 = frozen[i]
        fixed = dict(cohort=cohort, target_id=person, source_r2=source_r2, frozen_r2=frozen_r2)
        rows.append(dict(**fixed, budget=0, aligned_r2=frozen_r2, labeled_r2=frozen_r2,
                         correction_penalty="none", tuning_people=0))
        others = np.flatnonzero(people != person)
        assert len(others) == len(people)-1 and i not in others
        for j, budget in enumerate(BUDGETS):
            donor_validation = validation[others, j].mean(axis=0)
            near_min = np.flatnonzero(donor_validation <= donor_validation.min() + 1e-12)
            chosen = near_min[-1]  # Increasing shrinkage, ending at the exact no-correction candidate.
            alpha = ALPHAS[chosen]
            label = "none" if np.isinf(alpha) else f"{alpha:g}"
            aligned = test[i,j,-1]
            labeled = test[i,j,chosen]
            rows.append(dict(**fixed, budget=budget, aligned_r2=aligned, labeled_r2=labeled,
                             correction_penalty=label, tuning_people=len(others)))
            choices.append(dict(cohort=cohort, target_id=person, budget=budget,
                                correction_penalty=label, held_out_subject_index=i,
                                tuning_people=len(others), donor_mean_mse=donor_validation[chosen]))
            for k, candidate in enumerate(ALPHAS):
                losses.append(dict(cohort=cohort, donor_id=person, budget=budget,
                                   correction_penalty="none" if np.isinf(candidate) else f"{candidate:g}",
                                   validation_mse=validation[i,j,k]))
    return rows, choices, losses, errors


def summarize(people):
    rows = []
    for (cohort, n_rows), group in people.groupby(["cohort", "budget"], sort=False):
        group = group.sort_values("target_id")
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(group), size=(10000, len(group)))
        scores = {key:group[key+"_r2"].to_numpy() for key in ("source", "frozen", "aligned", "labeled")}
        measures = {key+"_r2":value for key,value in scores.items()}
        for first, second in (("aligned","frozen"), ("labeled","aligned"), ("labeled","frozen")):
            measures[first+"_minus_"+second] = scores[first] - scores[second]
        row = dict(cohort=cohort, budget=n_rows, n_people=len(group))
        for key, values in measures.items():
            lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            row[key] = values.mean()
            row[key+"_lo"] = lo
            row[key+"_hi"] = hi
            row[key+"_median"] = np.median(values)
            row[key+"_positive"] = int(np.count_nonzero(values>0))
        for key in ("aligned","labeled"):
            row[key+"_retention_pct"] = 100 * scores[key].mean() / scores["source"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def verify(people, summary, direct_errors):
    full = people.loc[people.budget.eq(CALIBRATION_N)]
    archive = pd.read_csv(transfer.ROOT/"results/portability-baseline/people.csv",
                          dtype={"target_id":str})
    cols = ["cohort","target_id","source_r2","untouched_r2","aligned_r2"]
    joined = full.merge(archive[cols], on=["cohort","target_id"], validate="one_to_one",
                        how="outer", indicator=True, suffixes=("", "_archived"))
    assert len(joined)==82 and joined._merge.eq("both").all()
    discrep = {}
    for key, archived in (("source_r2","source_r2_archived"),("frozen_r2","untouched_r2"),
                          ("aligned_r2","aligned_r2_archived")):
        delta = float(np.abs(joined[key]-joined[archived]).max())
        np.testing.assert_allclose(joined[key], joined[archived], rtol=0, atol=1e-11)
        discrep[key] = delta
    originals = pd.read_csv(transfer.ROOT/"results/portability-baseline/summary.csv")
    present = summary.loc[summary.budget.eq(CALIBRATION_N)].merge(originals, on="cohort")
    for key, archived in (("aligned_minus_frozen","delta_r2"),
                          ("aligned_minus_frozen_lo","delta_lo"),("aligned_minus_frozen_hi","delta_hi")):
        np.testing.assert_allclose(present[key],present[archived],rtol=0,atol=1e-11)
    return dict(participants=len(joined), total_participant_budgets=len(people),
                fresh_stream_roles=dict(source_train=0, calibration=219, validation_other_people=1, test=61),
                zero_budget_arms_equal=True, maximum_absolute_archived_errors=discrep,
                augmented_least_squares_checks=len(direct_errors),
                max_independent_absolute_score_or_mse_error=max(direct_errors),
                donor_validation_excludes_target=True, bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_draws=10000)


def main():
    start = perf_counter()
    people, choices, losses, direct = [], [], [], []
    for cohort in ("nki", "bnu1"):
        print(f"Evaluating {cohort}", flush=True)
        for collector, values in zip((people,choices,losses,direct),select_paths(cohort)):
            collector.extend(values)
    people = pd.DataFrame(people)
    summary = summarize(people)
    checks = verify(people, summary, direct)
    OUT.mkdir(parents=True, exist_ok=True)
    people.to_csv(OUT/"people.csv", index=False)
    summary.to_csv(OUT/"summary.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT/"penalty_selection.csv", index=False)
    pd.DataFrame(losses).to_csv(OUT/"validation_losses.csv.gz", index=False,
                                compression={"method":"gzip","mtime":0})
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(summary[["cohort","budget","frozen_r2","aligned_r2","labeled_r2","labeled_minus_aligned",
                   "labeled_minus_aligned_lo","labeled_minus_aligned_hi"]].to_string(index=False))
    print(f"Finished in {perf_counter()-start:.1f} seconds")


if __name__ == "__main__":
    main()
