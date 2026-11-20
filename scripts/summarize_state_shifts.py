"""Compare each graph substitution with source-state noise matched to its RMS displacement."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / 'results/portability-path'
KEYS = ['cohort', 'source_set', 'rung', 'task', 'alpha']
METRICS = ['shift_pct', 'source_r2', 'target_r2', 'isotropic_r2', 'graph_minus_iso',
           'actual_output_shift', 'expected_isotropic_loss', 'error_cross_term']


def participant_scores():
    visits = pd.read_csv(OUT / 'shift_visit_pairs.csv.gz', dtype={'source_id': str})
    technical = pd.read_csv(OUT / 'shift_technical_pairs.csv.gz', dtype={'source_id': str})
    first = technical.loc[technical.source_visit.eq(1) & technical.source_half.eq('odd')]
    visits['source_set'] = 'first_visit'
    technical = pd.concat([technical.assign(source_set='all_visit_halves'),
                           first.assign(source_set='first_visit')], ignore_index=True)
    changed = technical.loc[technical.rung.eq('numerical_perturbation') & technical.identical_offdiag.eq(0)]
    pairs = pd.concat([visits, technical, changed.assign(rung='numerical_changed_only')], ignore_index=True)
    pairs['shift_pct'] = 100 * pairs.delta_fraction
    pairs['graph_minus_iso'] = pairs.target_r2 - pairs.isotropic_r2
    if not np.isfinite(pairs[METRICS]).all().all():
        raise ValueError('Non-finite shift or accuracy')
    np.testing.assert_allclose(pairs.source_r2 - pairs.target_r2, pairs.actual_output_shift + pairs.error_cross_term,
                               rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(pairs.source_r2 - pairs.isotropic_r2, pairs.expected_isotropic_loss,
                               rtol=1e-7, atol=1e-11)
    # Equalize the original source cells before combining within each person.
    keys = KEYS + ['source_id']
    cells = pairs.groupby(keys + ['source_visit','source_half'])[METRICS].mean()
    sizes = cells.groupby(keys).size().rename('n_cells').reset_index()
    if not sizes.n_cells.eq(np.where(sizes.source_set.eq('first_visit'), 1, 4)).all():
        raise ValueError('Incomplete source positions')
    return cells.groupby(keys).mean().reset_index()


def summarize(people):
    rows = []
    for key, block in people.groupby(KEYS):
        block = block.sort_values('source_id')
        n = len(block)
        if n != {'nki':25, 'bnu1':57}[key[0]]:
            raise ValueError(f'Incomplete participants in {key}')
        samples = np.random.default_rng(20261120).integers(n, size=(10000,n))
        row = dict(zip(KEYS,key), n_people=n)
        for metric in METRICS:
            values = block[metric].to_numpy()
            row[metric] = values.mean()
            row[metric+'_median'] = np.median(values)
            row[metric+'_lo'], row[metric+'_hi'] = np.quantile(values[samples].mean(axis=1), [.025,.975])
        row['n_graph_positive'] = int(block.target_r2.gt(0).sum())
        row['n_isotropic_positive'] = int(block.isotropic_r2.gt(0).sum())
        row['n_iso_positive_graph_nonpositive'] = int((block.isotropic_r2.gt(0) & block.target_r2.le(0)).sum())
        row['n_graph_positive_iso_nonpositive'] = int((block.target_r2.gt(0) & block.isotropic_r2.le(0)).sum())
        row['n_graph_lower_than_isotropic'] = int(block.graph_minus_iso.lt(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(people):
    primary = people.loc[people.task.eq('linear') & people.alpha.eq(.001) & people.source_set.eq('first_visit')]
    fig, (size_ax, weak_ax, selected_ax) = plt.subplots(1, 3, figsize=(13.4, 4.8),
                                                       gridspec_kw={'width_ratios':[1.09,1,1]})
    colors = {'nki':'#276d83', 'bnu1':'#aa5139'}
    conditions = [('nki','numerical_perturbation','Numerical rerun'), ('nki','split_half','Other direction half'),
                  ('nki','between_visit','NKI repeat visit'), ('bnu1','between_visit','Beijing repeat visit')]
    size_ax.axvline(1, color='#695e53', lw=1.1, ls='--', label='Original 1% noise level')
    for y, (cohort, rung, label) in enumerate(conditions):
        block = primary.loc[primary.cohort.eq(cohort) & primary.rung.eq(rung)]
        jitter = np.random.default_rng(417+y).uniform(-.145,.145,len(block))
        size_ax.scatter(block.shift_pct, y+jitter, s=14, color=colors[cohort], alpha=.47, linewidths=0)
        size_ax.scatter(block.shift_pct.median(), y, s=55, marker='D', edgecolors='white',
                        linewidths=.9, facecolor='#202f36', zorder=3, label='Median' if y==0 else None)
    size_ax.set_yticks(range(4), [c[2] for c in conditions])
    size_ax.set_ylim(3.6,-.55)
    size_ax.set_xscale('symlog', linthresh=1)
    if (primary.shift_pct < 0).any():
        raise ValueError('Negative state RMS')
    right = max(120,primary.shift_pct.max()*1.15)
    size_ax.set_xlim(0,right)
    ticks = [t for t in (0,1,10,100,1000) if t<=right]
    size_ax.set_xticks(ticks,[str(t) for t in ticks])
    size_ax.set_xlabel('State change (% of source training RMS)')
    size_ax.set_title('How far did the states move?', loc='left', fontsize=12)
    size_ax.grid(axis='x', alpha=.16)
    size_ax.legend(loc='upper left',bbox_to_anchor=(0,-.18), fontsize=8, frameon=False, ncol=2)

    visits = people.loc[people.task.eq('linear') & people.rung.eq('between_visit')]
    for ax, weak, title in [(weak_ax,True,'Repeat visit at α = 0.001'),
                            (selected_ax,False,'Transfer-selected penalty')]:
        repeats = visits.loc[visits.alpha.eq(.001) == weak]
        vals = repeats[['target_r2','isotropic_r2']].to_numpy()
        lower = float(vals.min())*1.1
        upper = float(vals.max())+.025
        if weak:
            upper = 1.15
            ax.set_xscale('symlog', linthresh=1)
            ax.set_yscale('symlog', linthresh=1)
            power = int(np.floor(np.log10(-lower)))
            ticks = [-10**k for k in range(power,0,-1)]+[-1,0,1]
            for axis in [ax.xaxis,ax.yaxis]:
                axis.set_ticks(ticks, [str(t).replace('-','−') for t in ticks])
        ax.plot([lower,upper],[lower,upper],ls='--',color='#776e65',lw=.95)
        ax.axhline(0,color='#777777',lw=.65)
        ax.axvline(0,color='#777777',lw=.65)
        for cohort, name, marker in [('nki','NKI','o'),('bnu1','Beijing','^')]:
            group = repeats.loc[repeats.cohort.eq(cohort)]
            ax.scatter(group.isotropic_r2,group.target_r2,s=22,marker=marker,
                       color=colors[cohort],alpha=.76,linewidths=0,label=name)
        ax.set(xlim=(lower,upper),ylim=(lower,upper),xlabel='Expected R² under matched noise',
               ylabel='Repeat-scan R²', aspect='equal')
        ax.set_title(title, loc='left', fontsize=12)
    weak_ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8,frameon=False,ncol=2)
    for ax in (size_ax,weak_ax,selected_ax):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8.7)
    fig.suptitle('Measured wiring changes and size-matched state noise', x=.065, ha='left', fontsize=15)
    fig.subplots_adjust(left=.12,right=.985,bottom=.28,top=.83,wspace=.32)
    fig.text(.065,.05,'Linear recall with first-visit sources. Each point is one participant. Dashed diagonals mark equal scores. Transfer-selected α was 10 in NKI and 1 in Beijing.', fontsize=8.5)
    fig.savefig(OUT/'shift_matched_noise.png',dpi=170,facecolor='white')
    fig.savefig(OUT/'shift_matched_noise.pdf',facecolor='white',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__=='__main__':
    people = participant_scores()
    people.to_csv(OUT/'shift_people.csv',index=False)
    summarize(people).to_csv(OUT/'shift_summary.csv',index=False)
    plot(people)
