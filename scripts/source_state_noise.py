"""Evaluate frozen source readouts under independent additive observation noise."""
import json

from run_visit_portability_path import ALPHA, OUT, TASKS, TestDesign, fit_source, fresh_test, pilot, read_graphs, transfer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTIONS = np.r_[0., 10. ** np.arange(-8, -1)]
MODE_THRESHOLD = 1e-8


def task_values(outputs):
    """Reduce an array's last (eleven-outcome) dimension using study task weights."""
    return np.stack((outputs[..., :-1].mean(axis=-1), outputs[..., -1]), axis=-1)


def mode_diagnostics(model, weights, variance):
    # Sensitivity in R² units weights each outcome's slope norm by inverse target variance.
    # Training-state variance in the same basis also weights each mode by its eigenvalue.
    projected = np.einsum('mi,mpo->ipo', model.vec, weights)
    squared = projected**2 / variance
    low = model.val / model.scale < MODE_THRESHOLD
    np.testing.assert_allclose(squared.sum(axis=0), (weights**2).sum(axis=0) / variance, rtol=1e-12)
    sensitivity = task_values(squared.sum(axis=0))
    fitted_variance = squared * model.val[:, None, None]

    def fraction(values, total):
        numerator = task_values(values[low].sum(axis=0))
        return np.divide(numerator, total, out=np.full_like(total, np.nan), where=total > 0)

    return (low, sensitivity, fraction(squared, sensitivity),
            fraction(fitted_variance, task_values(fitted_variance.sum(axis=0))))


def injection_check(cohort, subject, index, model, weights, x, y, analytic):
    """Check analytic MSE inflation using actual independent Gaussian state perturbations."""
    test_steps, n_features = x.shape
    records = []
    for alpha_index in (0, 12, 14):
        weight = weights[:, alpha_index, :]
        residual = (x - model.mx) @ weight + model.my - y
        # Positive and negative copies cancel the cross term repeat by repeat.
        # Eight independent draws estimate the expected squared noise contribution.
        empirical = []
        for repeat in range(8):
            rng = pilot.rng(916, 0 if cohort == 'nki' else 1, index, repeat)
            delta = rng.normal(size=(test_steps, n_features)) * np.sqrt(model.scale) * 1e-3
            clean_mse = (residual**2).mean(axis=0)
            positive = (x + delta - model.mx) @ weight + model.my - y
            negative = (x - delta - model.mx) @ weight + model.my - y
            extra_mse = (np.mean(positive**2, axis=0) + np.mean(negative**2, axis=0)) / 2 - clean_mse
            empirical.append(task_values(extra_mse / y.var(axis=0)))
        observed = np.mean(empirical, axis=0)
        expected = analytic[alpha_index] * 1e-6
        np.testing.assert_allclose(observed, expected, rtol=.06, atol=1e-13)
        for k, task in enumerate(TASKS):
            records.append(dict(cohort=cohort, subject=str(subject), alpha=float(ALPHA[alpha_index]),
                                task=task, noise_fraction=.001, empirical_loss=float(observed[k]),
                                expected_loss=float(expected[k]), relative_error=float(abs(observed[k]/expected[k]-1))))
    return records


def cohort_scores(cohort):
    graphs, ids, _ = read_graphs(cohort)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    scores, modes, checks = [], [], []
    for i, (graph, subject) in enumerate(zip(graphs[0], ids)):
        net = pilot.reservoir(graph, transfer.GAIN)
        model, weights = fit_source(net, mapping, 10000 + i)
        x, y = fresh_test(net, mapping, 10000 + len(ids) + i)
        clean = TestDesign.from_data(x, y).score(model, weights)
        low, sensitivity, low_sensitivity, low_fit = mode_diagnostics(model, weights, y.var(axis=0))
        loss_unit = sensitivity * model.scale
        if i in (0, len(ids)//2):
            checks.extend(injection_check(cohort, subject, i, model, weights, x, y, loss_unit))
        for t, task in enumerate(TASKS):
            modes.append(pd.DataFrame(dict(cohort=cohort, source_id=str(subject), task=task,
                alpha=ALPHA, state_rms=np.sqrt(model.scale), n_modes=len(low), n_low_modes=low.sum(),
                low_state_variance_share=model.val[low].sum()/model.val.sum(),
                low_slope_energy_share=low_sensitivity[:, t], low_fitted_variance_share=low_fit[:, t],
                unit_noise_r2_loss=loss_unit[:, t])))
            for frac in FRACTIONS:
                scores.append(pd.DataFrame(dict(cohort=cohort, source_id=str(subject), task=task,
                    alpha=ALPHA, noise_fraction=frac, noise_sd=frac*np.sqrt(model.scale),
                    clean_source_r2=clean[:, t], expected_source_r2=clean[:, t] - frac**2*loss_unit[:, t])))
    return pd.concat(scores, ignore_index=True), pd.concat(modes, ignore_index=True), checks


def draw(summary):
    # A common display scale preserves low but positive performance as well as the upper range.
    shown = [(0., 'Clean', '#202b2f', '-'), (1e-6, '0.0001%', '#448979', '--'),
             (1e-4, '0.01%', '#2671a7', '-.'), (.001, '0.1%', '#9a5128', '--'),
             (.01, '1%', '#90486b', '-')]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=True)
    for row, cohort in enumerate(('nki', 'bnu1')):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            block = summary.loc[summary.cohort.eq(cohort) & summary.task.eq(task) & summary.alpha.le(1e4)]
            for frac, label, color, style in shown:
                line = block.loc[block.noise_fraction.eq(frac)].sort_values('alpha')
                ax.plot(np.log10(line.alpha), line.mean_r2, color=color, linestyle=style, lw=1.6,
                        marker='o', ms=2.8, label=label)
            ax.set_title(('NKI' if cohort == 'nki' else 'Beijing') + '  ·  ' +
                         ('Linear recall' if task == 'linear' else 'Three-lag product'), loc='left', fontsize=11)
            ax.set(ylim=(-.04, 1.03), xlim=(-14.25, 4.25), xticks=np.arange(-14, 5, 3))
            ax.grid(axis='y', alpha=.2)
            ax.axhline(0, lw=.65, color='#777777')
            ax.spines[['top', 'right']].set_visible(False)
            if col == 0:
                ax.set_ylabel('Mean expected source R²')
            if row == 1:
                ax.set_xlabel('log₁₀ relative ridge penalty')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(1.15, 1.47), ncol=len(shown),
                      title='Noise SD (% of source training state RMS)', frameon=False, fontsize=9)
    fig.suptitle('Frozen decoders under source-state observation noise', x=.11, y=.985, ha='left', fontsize=15)
    fig.text(.11, .055, 'Points show the original 19 penalties. Tables include the extended path and all negative scores.\nNoise is added to observed states at test time on the same graph.', va='top', fontsize=8.5)
    fig.subplots_adjust(left=.11, right=.97, top=.78, bottom=.17, hspace=.32, wspace=.18)
    for extension in ('png', 'pdf'):
        opts = {'metadata': {'CreationDate': None, 'ModDate': None}} if extension == 'pdf' else {'dpi': 170}
        fig.savefig(OUT / f'noise_source_paths.{extension}', facecolor='white', **opts)
    plt.close(fig)


def main():
    frames, mode_frames, injection = [], [], []
    for cohort in ('nki', 'bnu1'):
        frame, modes, checks = cohort_scores(cohort)
        frames.append(frame)
        mode_frames.append(modes)
        injection.extend(checks)
    people = pd.concat(frames, ignore_index=True)
    modes = pd.concat(mode_frames, ignore_index=True)
    key = ['cohort', 'source_id', 'task', 'alpha']
    original = pd.read_csv(OUT/'visit_pairs.csv.gz', usecols=key+['rung', 'source_r2'], dtype={'source_id': str})
    clean = people.loc[people.noise_fraction.eq(0)].merge(original.loc[original.rung.eq('between_visit'), key+['source_r2']],
                                                          on=key, how='left', validate='one_to_one')
    assert len(clean) == 82*24*2
    np.testing.assert_allclose(clean.expected_source_r2, clean.source_r2, atol=1e-8, rtol=0)
    constant = people.loc[np.isinf(people.alpha)]
    np.testing.assert_array_equal(constant.clean_source_r2, constant.expected_source_r2)
    summary = people.groupby(['cohort','task','noise_fraction','alpha'], as_index=False).agg(
        n=('source_id','size'), mean_r2=('expected_source_r2','mean'), median_r2=('expected_source_r2','median'),
        n_positive=('expected_source_r2', lambda v: int(v.gt(0).sum())),
        minimum_absolute_noise_sd=('noise_sd','min'), maximum_absolute_noise_sd=('noise_sd','max'))
    envelope = summary.loc[summary.groupby(['cohort','task','noise_fraction']).mean_r2.idxmax()].copy()
    assert len(envelope) == 4*len(FRACTIONS)
    summary.to_csv(OUT/'noise_summary.csv', index=False)
    envelope.to_csv(OUT/'noise_envelope.csv', index=False)
    mode_summary = modes.groupby(['cohort','task','alpha'], as_index=False).agg(
        n=('source_id','size'), median_low_slope_energy_share=('low_slope_energy_share','median'),
        median_low_fitted_variance_share=('low_fitted_variance_share','median'),
        median_low_state_variance_share=('low_state_variance_share','median'),
        median_n_low_modes=('n_low_modes','median'), n_state_modes=('n_modes','first'))
    mode_summary.to_csv(OUT/'noise_modes.csv', index=False)
    compression = dict(method='gzip', mtime=0)
    people.to_csv(OUT/'noise_people.csv.gz', index=False, compression=compression)
    modes.to_csv(OUT/'noise_modes_people.csv.gz', index=False, compression=compression)
    info = dict(fractions=FRACTIONS.tolist(), calibration='Per-scan state RMS is sqrt(mean variance across readout features on the source training sequence). Absolute Gaussian noise SD is fraction times this RMS.',
                insertion='Zero-mean independent noise is added to each readout feature at test time, after the recurrence. Training and decoder weights remain fixed. No scan noise estimate is assumed.',
                quantity='Exact expectation of per-output test MSE = clean MSE + noise_variance times squared slope norm; convert separately to each output R2 with its unchanged test variance, then average delays.',
                envelope='Descriptive maximum of cohort mean expected fresh-source score over common tested penalties at each noise level. No noisy test fitting or tuning used for individual decoders.',
                spectrum=f'Training covariance eigenvalues below {MODE_THRESHOLD:g} times their across-feature mean. Energy fractions use squared slope per output divided by its test target variance, then pool over outputs. Fitted-score variance additionally weights squared slope by the corresponding eigenvalue.',
                absent_values='For the zero-slope constant model, slope-energy and fitted-variance fractions are undefined.',
                clean_archive_max_abs_error=float(abs(clean.expected_source_r2-clean.source_r2).max()), n_clean_checks=len(clean),
                injection_method='Four deterministic source scans, 3 penalties, 8 independent Gaussian realizations on 2,048 steps plus sign-reversed copies, compared with the analytic MSE increment.',
                injection_max_relative_error=max(e['relative_error'] for e in injection), injection_checks=injection)
    (OUT/'noise_checks.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    draw(summary)
    print(envelope[['cohort','task','noise_fraction','alpha','mean_r2']].to_string(index=False))
    print('Maximum source replay error', info['clean_archive_max_abs_error'], 'maximum injection relative error', info['injection_max_relative_error'])


if __name__ == '__main__':
    main()
