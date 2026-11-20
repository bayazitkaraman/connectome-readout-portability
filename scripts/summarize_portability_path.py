"""Summarize matched source/target ridge paths, with people as the unit of analysis."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/portability-path'
KEY = ['cohort', 'task', 'rung', 'alpha']
SCORES = ['source_r2', 'target_r2']
PATH = np.r_[10.0 ** np.arange(-14, 9), np.inf]
SIZES = {'nki': 25, 'bnu1': 57}
STYLE = {
    'numerical_perturbation': ('Numerical rerun', '#16755c', 'o'),
    'split_half': ('Other direction half', '#2775aa', 's'),
    'between_visit': ('Own repeat visit', '#b4471b', 'D'),
    'stranger': ('Other person (mean)', '#755587', '^'),
}


def aggregate():
    frames = []
    dtypes = {'target_id': str, 'source_id': str}
    visits = pd.read_csv(OUT / 'visit_pairs.csv.gz', dtype=dtypes)
    rungs = pd.read_csv(OUT / 'rung_pairs.csv.gz', dtype=dtypes)
    # An MCA cell is one original visit and direction half. Equalize these
    # four cells even in the sensitivity analysis where replicate counts vary.
    for name, raw, is_nki_cell in [
        ('visits', visits, False), ('nki_rungs', rungs, True),
        ('changed', rungs.loc[rungs.rung.eq('numerical_perturbation') &
                              rungs.identical_offdiag.eq(0)].copy(), True),
    ]:
        if name == 'changed':
            raw['rung'] = 'numerical_changed_only'
        if not np.isfinite(raw[SCORES]).all().all():
            raise ValueError(f'Non-finite scores in {name}')
        subject = KEY + ['target_id']
        if is_nki_cell:
            cells = raw.groupby(subject + ['source_visit', 'source_half'])[SCORES].mean()
            if not cells.groupby(subject).size().eq(4).all():
                raise ValueError('Incomplete original visit/direction cells')
            people = cells.groupby(subject).mean().reset_index()
        else:
            people = raw.groupby(subject, as_index=False)[SCORES].mean()
        frames.append(people)
    people = pd.concat(frames, ignore_index=True)
    people['gap_r2'] = people.source_r2 - people.target_r2
    np.testing.assert_allclose(people.loc[np.isinf(people.alpha), 'gap_r2'], 0, atol=1e-12)
    summary = []
    for key, block in people.groupby(KEY, sort=True):
        cohort = key[0]
        block = block.sort_values('target_id')
        n = len(block)
        if n != SIZES[cohort] or block.target_id.nunique() != n:
            raise ValueError(f'Unexpected participant count in {key}')
        draws = np.random.default_rng(20261120).integers(n, size=(10000, n))
        row = dict(zip(KEY, key), n_people=n)
        for metric in SCORES + ['gap_r2']:
            values = block[metric].to_numpy()
            row[metric] = values.mean()
            row[metric + '_lo'], row[metric + '_hi'] = np.quantile(values[draws].mean(axis=1), [.025, .975])
            row[metric + '_median'] = np.median(values)
        summary.append(row)
    summary = pd.DataFrame(summary)
    for _, block in summary.groupby(['cohort', 'task', 'rung']):
        np.testing.assert_allclose(np.sort(block.alpha), PATH, rtol=1e-12, atol=0)
    return summary


def check_donor_contrast():
    raw = pd.read_csv(ROOT / 'results/pipeline-robustness/task_comparison_people.csv', dtype={'target_id': str})
    rows = []
    for cohort, group in raw.loc[raw.task.eq('linear')].groupby('cohort'):
        group = group.sort_values('target_id')
        diff = (group.nearest_r2 - group.own_r2).to_numpy()
        draws = np.random.default_rng(20261120).integers(len(group), size=(10000, len(group)))
        lo, hi = np.quantile(diff[draws].mean(axis=1), [.025, .975])
        rows.append(dict(cohort=cohort, n=len(diff), nearest=group.nearest_r2.mean(), own=group.own_r2.mean(),
                         typical=group.mean_other_r2.mean(), pooled=group.pooled_r2.mean(),
                         nearest_minus_own=diff.mean(), lo=lo, hi=hi, positive=int((diff > 0).sum())))
    return pd.DataFrame(rows)


def plot(summary, full):
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.1), sharex=True, sharey=True)
    y_min = min(-1, summary.target_r2.min() * 1.4) if full else -0.025
    for row, cohort in enumerate(['nki', 'bnu1']):
        for col, task in enumerate(['linear', 'nonlinear']):
            ax = axes[row, col]
            block = summary.loc[summary.cohort.eq(cohort) & summary.task.eq(task)]
            if full:
                ax.set_yscale('symlog', linthresh=1)
            ax.plot([0, 1], [0, 1], '--', color='#818181', lw=0.85, zorder=1)
            for rung, (_, color, marker) in STYLE.items():
                line = block.loc[block.rung.eq(rung)].sort_values('alpha')
                if line.empty:
                    continue
                ax.plot(line.source_r2, line.target_r2, marker=marker, ms=3.4, lw=1.4,
                        color=color, alpha=0.9, clip_on=True)
                # The primary linear operating point is a single penalty per cohort.
                if rung == 'between_visit' and task == 'linear':
                    primary = 10 if cohort == 'nki' else 1
                    chosen = line.loc[line.alpha.eq(primary)]
                    ax.scatter(chosen.source_r2, chosen.target_r2, marker='o', s=90,
                               facecolors='none', edgecolors='#222222', linewidth=1, zorder=5)
                    if not full:
                        point = chosen.iloc[0]
                        ax.annotate(f'α = {primary}', (point.source_r2, point.target_r2),
                                    xytext=(9, 7), textcoords='offset points', color=color, fontsize=8)
                if not full and task == 'linear' and rung in ('numerical_perturbation', 'split_half'):
                    alpha = 0.01 if rung == 'numerical_perturbation' else 1
                    point = line.loc[line.alpha.eq(alpha)].iloc[0]
                    ax.annotate(f'α = {alpha}', (point.source_r2, point.target_r2),
                                xytext=(4, 9), textcoords='offset points', color=color, fontsize=8)
            if not full:
                ax.set_aspect('equal', adjustable='box')
            ax.set(xlim=(-0.025, 1 if full else .60), ylim=(y_min, 1.015 if full else .60))
            if not full and task == 'nonlinear' and block.target_r2.max() < 0:
                ax.text(.05, .70, 'All mean target scores\nare below zero.', transform=ax.transAxes,
                        fontsize=9.5, color='#454545')
            ax.axhline(0, lw=0.55, color='#888888', zorder=0)
            ax.grid(axis='y', lw=0.5, alpha=0.22)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_title(('NKI' if cohort == 'nki' else 'Beijing') + '  ·  ' +
                         ('Linear recall' if task == 'linear' else 'Three-lag product'), loc='left', fontsize=11)
            if row == 1:
                ax.set_xlabel('Fresh-signal R² on source')
            if col == 0:
                ax.set_ylabel('Same decoder R² on target')
    if full:
        # Keep the positive band visible through a clearly labelled detail figure.
        max_power = max(2, int(np.floor(np.log10(abs(y_min)))))
        powers = np.arange(2, max_power + 1, 3)
        ticks = [-10.0**int(k) for k in powers[::-1]] + [-1, 0, 1]
        axes[0, 0].set_yticks(ticks, [rf'$-10^{{{k}}}$' for k in powers[::-1]] + ['−1', '0', '1'])
    legend = [Line2D([], [], color=color, marker=marker, markersize=4, label=label)
              for label, color, marker in STYLE.values()]
    fig.suptitle('Accuracy and portability across ridge penalties', x=0.11, ha='left', y=0.985, fontsize=16)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.53, 0.94), frameon=False, ncol=4, fontsize=9)
    caption = ('Full range. The vertical axis is logarithmic below −1 and linear between −1 and 1.' if full
               else 'Detail near and above zero. The companion figure shows every score, including large negative values.')
    fig.text(0.11, 0.066, caption + '\nStronger penalties lead toward the origin. The dashed line marks equal scores. Rings mark the selected linear repeat-scan penalties.',
             va='top', fontsize=8.6, linespacing=1.4)
    fig.subplots_adjust(left=.11, right=.97, top=.84, bottom=.16, hspace=.33, wspace=.19)
    basename = 'accuracy_portability_full' if full else 'accuracy_portability_detail'
    fig.savefig(OUT / f'{basename}.png', dpi=170, facecolor='white')
    fig.savefig(OUT / f'{basename}.pdf', facecolor='white',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    result = aggregate()
    result.to_csv(OUT / 'path_summary.csv', index=False)
    check_donor_contrast().to_csv(OUT / 'nearest_vs_own.csv', index=False)
    for full_range in [False, True]:
        plot(result, full_range)
