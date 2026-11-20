"""Measure repeat-scan state displacements and test size-matched isotropic readout noise."""
import json

from run_visit_portability_path import OUT, TASKS, fit_source, fresh_test, pilot, read_graphs, transfer
import numpy as np
import pandas as pd


def evaluate_shift(source_states, target_states, y, model, alphas):
    """Compare actual wiring displacement with independent isotropic noise of equal RMS."""
    delta = target_states - source_states
    delta_power = np.mean(delta**2)
    train_rms = np.sqrt(model.scale)
    variance = np.var(y, axis=0)
    rows = []
    for alpha in alphas:
        slopes = model.weights(alpha)
        residual = (source_states - model.mx) @ slopes + model.my - y
        output_shift = delta @ slopes
        source = 1 - np.mean(residual**2, axis=0) / variance
        target = 1 - np.mean((residual + output_shift)**2, axis=0) / variance
        amplification = np.mean(output_shift**2, axis=0) / variance
        isotropic_loss = delta_power * np.sum(slopes**2, axis=0) / variance
        cross = 2 * np.mean(residual * output_shift, axis=0) / variance
        np.testing.assert_allclose(source-target, amplification+cross, atol=1e-12, rtol=1e-12)
        for task, outputs in zip(TASKS, [slice(None, -1), slice(-1, None)]):
            rows.append(dict(
                task=task, alpha=alpha, train_state_rms=train_rms, delta_rms=np.sqrt(delta_power),
                delta_fraction=np.sqrt(delta_power)/train_rms,
                source_r2=source[outputs].mean(), target_r2=target[outputs].mean(),
                isotropic_r2=(source-isotropic_loss)[outputs].mean(),
                actual_output_shift=amplification[outputs].mean(),
                expected_isotropic_loss=isotropic_loss[outputs].mean(),
                error_cross_term=cross[outputs].mean()))
    return rows


def injected_noise_check(xs, xt, y, model, alphas, rows, cohort, index):
    """Verify the equal-energy expectation with explicit paired Gaussian draws."""
    sigma = np.sqrt(np.mean((xt-xs)**2))
    variance = np.var(y, axis=0)
    observed = {alpha: np.zeros(y.shape[1]) for alpha in alphas}
    trials = 16
    for repeat in range(trials):
        rng = pilot.rng(218, int(cohort=="bnu1"), index, repeat)
        noise = rng.normal(0, sigma, size=xs.shape)
        for alpha in alphas:
            plus = model.predict(xs+noise, alpha) - y
            minus = model.predict(xs-noise, alpha) - y
            observed[alpha] += np.mean(plus**2+minus**2, axis=0) / (2*trials)
    worst = 0.
    for row in rows:
        outputs = slice(None, -1) if row["task"]=="linear" else slice(-1, None)
        simulated_r2 = (1 - observed[row["alpha"]]/variance)[outputs].mean()
        relative_error = abs(simulated_r2-row["isotropic_r2"]) / row["expected_isotropic_loss"]
        worst = max(worst, relative_error)
    assert worst < .05
    return float(worst)


def cohort_pairs(cohort):
    graphs, ids, half = read_graphs(cohort)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    rows, injections = [], []
    alphas = (0.001, 10.0 if cohort == "nki" else 1.0)
    edge_index = np.triu_indices(graphs.shape[-1], 1)
    for i, subject in enumerate(ids):
        source_net = pilot.reservoir(graphs[0, i], transfer.GAIN)
        target_net = pilot.reservoir(graphs[1, i], transfer.GAIN)
        train_stream = 10000+i
        test_stream = train_stream+len(ids)
        model, _ = fit_source(source_net, mapping, train_stream)
        xs, y = fresh_test(source_net, mapping, test_stream)
        xt, y_repeat = fresh_test(target_net, mapping, test_stream)
        np.testing.assert_array_equal(y, y_repeat)
        identical = np.array_equal(graphs[0, i][edge_index], graphs[1, i][edge_index])
        scores = evaluate_shift(xs, xt, y, model, alphas)
        if i in (0, len(ids)//2):
            injections.append(injected_noise_check(xs, xt, y, model, alphas, scores, cohort, i))
        for score in scores:
            rows.append(dict(cohort=cohort, rung="between_visit", target_id=subject, source_id=subject,
                             source_visit=1, target_visit=2, source_half=half, target_half=half,
                             run=-1, identical_offdiag=int(identical), **score))
    return pd.DataFrame(rows), max(injections)


def check_against_archives(frame):
    keys = ["cohort", "target_id", "task", "alpha"]
    original = pd.read_csv(OUT/"visit_pairs.csv.gz", usecols=keys+["rung","source_r2","target_r2"],
                           dtype={"target_id": str})
    original = original.loc[original.rung.eq("between_visit"), keys+["source_r2","target_r2"]]
    joined = frame.merge(original, on=keys, how="left", suffixes=("", "_archived"), validate="one_to_one")
    errors = {}
    for metric in ["source_r2","target_r2"]:
        actual, expected = joined[metric], joined[metric+"_archived"]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        errors[metric] = float((actual-expected).abs().max())
    # The old state-noise path evaluated fixed scales with an independently implemented
    # calculation. Its loss coefficient scales with the square of the noise SD.
    noise = pd.read_csv(OUT/"noise_modes_people.csv.gz", dtype={"source_id": str})
    keys = ["cohort","task","source_id","alpha"]
    joined = frame.merge(noise[keys+["unit_noise_r2_loss"]], on=keys, how="left", validate="one_to_one")
    expected = joined.unit_noise_r2_loss * joined.delta_fraction**2
    np.testing.assert_allclose(joined.expected_isotropic_loss, expected, rtol=1e-12, atol=1e-12)
    errors["isotropic_loss"] = float((joined.expected_isotropic_loss-expected).abs().max())
    return errors


def main():
    results = [cohort_pairs(cohort) for cohort in ("nki","bnu1")]
    frame = pd.concat([rows for rows, _ in results], ignore_index=True)
    assert len(frame) == (25+57)*2*2
    errors = check_against_archives(frame)
    frame.to_csv(OUT/"shift_visit_pairs.csv.gz", index=False,
                 compression={"method":"gzip","mtime":0})
    checks = dict(pairs=82, rows=len(frame), archive_max_abs_difference=errors,
                  injection="Four fixed participants, both tasks and penalties, 16 independent Gaussian realizations per participant at their measured repeat-scan RMS, with sign-reversed partners.",
                  injection_max_relative_error_in_loss=max(worst for _, worst in results),
                  inputs="Same four independent role-61 streams on the original and repeat graph of each participant.",
                  units="RMS of raw difference on observed readout states divided by training-state RMS (square root of mean centered feature variance).",
                  matched_isotropic="Zero-mean independent noise on each source readout coordinate/time point with expected mean square equal to the observed graph displacement mean square.",
                  decomposition="target_r2 = source_r2 - actual_output_shift - error_cross_term; isotropic_r2 = source_r2 - expected_isotropic_loss. Normalize per output by test label variance before averaging delays.")
    (OUT/"shift_visit_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
