"""Score deterministic NKI input-MCA reruns with a frozen, fixed-penalty readout."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nki_transfer as transfer
from verify_nki_calibration import adjacency
from verify_nki_split_half import file_hash
from verify_results import SingularRidge, fixtures, join, stream
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/numerical-repeat'
HDF_MD5 = 'f457d4a860d79654672143c1ace95ee1'
ROLE = 61
PENALTY = 10.
PILOT = transfer.pilot
KINDS = ['reference', 'input_mca', 'direction_half', 'other_visit']
METRICS = ['similarity', 'top1', 'own_r2', 'other_r2', 'own_gain', 'r2_loss', 'pred_rmse']


def fetch_data(hdf):
    if hdf.stat().st_size != 233049104 or file_hash(hdf, 'md5') != HDF_MD5:
        raise ValueError('HDF size or checksum differs from the public release.')
    # The fixed-format HDF uses object blocks. Read only the checksum-verified original.
    table = pd.read_hdf(hdf, key='graphs')
    det = table.loc[table.pipeline.eq('det'), ['subject','session','directions','simulation','seed','graph']]
    levels = ['subject','session','directions','simulation']
    assert len(table) == 4200 and len(det) == 2100 and set(det.seed) == {42}
    assert not det.duplicated(levels).any()
    names = ['ref']+[f'sim-{k}' for k in range(20)]
    assert set(det.simulation) == set(names)
    graphs = det.set_index(levels).graph.sort_index()
    with np.load(transfer.DEFAULT_DATA, allow_pickle=False) as z:
        subjects, visits, halves = z['subjects'], z['session_labels'], z['halves']
        ref = z['weights']
    assert np.array_equal(halves, ['odd','even']) and ref.shape == (2,2,25,83,83)
    for i, subject in enumerate(subjects):
        for v in range(2):
            for h, half in enumerate(halves):
                group = graphs.loc[(subject, visits[v,i], half)]
                assert set(group.index) == set(names)
                np.testing.assert_array_equal(ref[h,v,i], group.loc['ref'])
                for graph in group:
                    assert graph.shape == (83,83) and np.array_equal(graph,graph.T)
                    assert np.isfinite(graph).all() and np.min(graph) >= 0
    return subjects, visits, halves, ref, graphs


def reservoir(raw):
    a = np.array(raw,dtype=float)
    np.fill_diagonal(a,0.)
    return PILOT.reservoir(a, transfer.GAIN)


def standardized_edges(raw):
    v = raw[np.triu_indices(len(raw),1)].astype(float)
    v -= v.mean()
    return v / np.linalg.norm(v)


def reference_models(bank, mapping, visit):
    coeffs = []
    for i, graph in enumerate(bank):
        ident = 10000+len(bank)*visit+i
        xy = PILOT.simulate(reservoir(graph),mapping,ident,0,PILOT.FIT_N)
        model = PILOT.RidgeFamily.from_design(xy)
        slopes = model.weights(PENALTY)
        coeffs.append(np.vstack([model.my-model.mx@slopes, slopes]))
    return np.stack(coeffs)


def score_graph(graph, edges, coeffs, i, ident, mapping):
    net = reservoir(graph)
    x,y = PILOT.combine([PILOT.simulate(net,mapping,ident,ROLE,512,k) for k in range(4)])
    # Use second moments for the full donor bank and retain explicit own predictions.
    x = np.column_stack([np.ones(len(x)), x])
    g, c, y2 = (x.T@x)/len(x), (x.T@y)/len(x), np.mean(y*y,axis=0)
    mse = y2 + (coeffs * ((g@coeffs)-2*c)).sum(axis=1)
    scores = (1.-mse/y.var(axis=0)).mean(axis=1)
    pred = x@coeffs[i]
    direct = float((1.-((pred-y)**2).mean(axis=0)/y.var(axis=0)).mean())
    np.testing.assert_allclose(direct,scores[i],atol=1e-12,rtol=0.)
    sims = edges @ standardized_edges(graph)
    tied = np.isclose(sims,sims[i],rtol=0.,atol=1e-12)
    rank = 1+int(((sims>sims[i]) & ~tied).sum())
    other = float(np.delete(scores,i).mean())
    return dict(similarity=sims[i], rank=rank, tied=int(tied.sum()),
                top1=int(rank==1 and tied.sum()==1), own_r2=direct, other_r2=other,
                own_gain=direct-other),pred


def independent(source,graph,i,v,own,other,bank):
    inputs = np.zeros(83)
    places = stream(10).choice(83,8,replace=False)
    inputs[places[:4]], inputs[places[4:]] = .5,-.5
    ident = 10000+25*v+i
    test = join([fixtures(adjacency(graph),inputs,ident,61,512,k) for k in range(4)])
    x,y = test
    independent_scores = []
    for d, raw in enumerate(bank):
        model = SingularRidge(fixtures(adjacency(raw),inputs,10000+25*v+d,0,2352))
        independent_scores.append(float((1.-((model.predict(x,10.)-y)**2).mean(0)/y.var(0)).mean()))
    anatomical = np.corrcoef(source[np.triu_indices(83,1)],graph[np.triu_indices(83,1)])[0,1]
    return max(abs(independent_scores[i]-own),abs(np.delete(independent_scores,i).mean()-other)), anatomical


def collect(subjects, visits, halves, ref, graphs):
    out, errors, correlations = [],[],[]
    mapping = PILOT.input_map(83,key=10)
    for v in range(2):
        for h, half in enumerate(halves):
            bank = ref[h,v]
            fitted = reference_models(bank,mapping,v)
            edge = np.stack([standardized_edges(g) for g in bank])
            for i,person in enumerate(subjects):
                key = (person,visits[v,i],half)
                variants = [('reference',-1,bank[i])]
                variants.extend(('input_mca',k,graphs.loc[(*key,f'sim-{k}')]) for k in range(20))
                variants.extend([('direction_half',-1,ref[1-h,v,i]),('other_visit',-1,ref[h,1-v,i])])
                ref_score = ref_pred = None
                for kind, k, graph in variants:
                    result, pred = score_graph(graph,edge,fitted,i,10000+25*v+i,mapping)
                    if kind=='reference':
                        ref_score, ref_pred = result['own_r2'],pred
                        assert result['top1'] == 1 and abs(result['similarity']-1.)<1e-12
                    result['r2_loss'] = ref_score-result['own_r2']
                    result['pred_rmse'] = float(np.sqrt(np.mean((pred-ref_pred)**2)))
                    tri = np.triu_indices(83,1)
                    result['identical_offdiag'] = int(np.array_equal(bank[i][tri],graph[tri]))
                    result['edge_sha256'] = hashlib.sha256(np.asarray(graph[tri],dtype='<f8').tobytes()).hexdigest()
                    if i in (0,12,24) and ((v==0 and h==0) or i==0) and k in (-1,0,19):
                        mse_err, rho = independent(bank[i],graph,i,v,result['own_r2'],result['other_r2'],bank)
                        errors.append(mse_err)
                        correlations.append(abs(rho-result['similarity']))
                    out.append(dict(person=person,visit=v+1,half=half,source_session=visits[v,i],
                                    condition=kind,run=k,**result))
            print(f'Completed visit {v+1}, {half}',flush=True)
    return pd.DataFrame(out), errors, correlations


def summaries(rows):
    condition = ['person','visit','half','condition']
    cells = rows.groupby(condition,as_index=False)[METRICS].mean()
    ref = rows.query("condition=='reference'")
    assert len(rows)==2300 and len(cells)==400 and ref.pred_rmse.eq(0).all() and ref.r2_loss.eq(0).all()
    mca = rows.query("condition=='input_mca'")
    run_sd = mca.groupby(['person','visit','half'])[METRICS].std().add_suffix('_within_cell_run_sd').reset_index()
    n_distinct = rows[rows.condition=='input_mca'].groupby('person').size()
    assert n_distinct.eq(80).all()
    cells = cells.merge(run_sd.assign(condition='input_mca'),on=condition,how='left',validate='one_to_one')
    means = cells.groupby(['person','condition'],as_index=False)[METRICS].mean()
    summary = []
    def one(frame,visit,half):
        for cond, sub in frame.groupby('condition'):
            assert len(sub) == 25 and sub.person.nunique()==25
            sample = np.random.default_rng(20261119).integers(25,size=(10000,25))
            for metric in METRICS:
                a = sub[metric].to_numpy()
                b = a[sample].mean(axis=1)
                summary.append(dict(visit=visit,half=half,condition=cond,metric=metric,n_people=len(a),
                                    mean=a.mean(),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975),
                                    person_sd=a.std(ddof=1),minimum=a.min(),maximum=a.max()))
    for (v,h),f in cells.groupby(['visit','half']):
        one(f,v,h)
    one(means,'all','all')
    differences = []
    for (visit,half), frame in [((v,h),f) for (v,h),f in cells.groupby(['visit','half'])]+[(('all','all'),means)]:
        for metric in ['own_r2','r2_loss','similarity','own_gain']:
            pivot = frame.pivot(index='person',columns='condition',values=metric)
            for a,b in [('reference','input_mca'),('input_mca','direction_half'),('input_mca','other_visit'),
                        ('direction_half','other_visit')]:
                d=(pivot[a]-pivot[b]).to_numpy()
                ix=np.random.default_rng(20261119).integers(25,size=(10000,25))
                lo,hi=np.quantile(d[ix].mean(axis=1),[.025,.975])
                differences.append(dict(visit=visit,half=half,metric=metric,first=a,second=b,
                                        mean=d.mean(),ci_low=lo,ci_high=hi,n_positive=int(np.sum(d>0))))
    return cells,means,pd.DataFrame(summary),pd.DataFrame(differences)


def main(hdf):
    start=perf_counter()
    subjects,visits,halves,ref,graphs = fetch_data(hdf)
    rows,errors,correlations=collect(subjects,visits,halves,ref,graphs)
    cells,people,summary,diffs=summaries(rows)
    archive=pd.read_csv(ROOT/'results/transfer-similarity/source_self_fixed10_subjects.csv',dtype={'target_id':str})
    archive=archive.query("cohort=='nki' and src_visit==target_visit and src_half!=target_half")
    check=rows[rows.condition=='direction_half'].merge(archive,left_on=['person','visit','half'],
                right_on=['target_id','src_visit','src_half'],validate='one_to_one')
    assert len(check) == 100
    archive_error=max(abs(check.own_r2-check.transfer_target_r2).max(),
                      abs(check.r2_loss-check.frozen_wiring_drop).max())
    assert max(errors)<1e-11 and max(correlations)<1e-12 and archive_error<1e-11
    for file,df in [('runs',rows),('cells',cells),('people',people),('summary',summary),('contrasts',diffs)]:
        df.to_csv(OUT/(file+'.csv'),index=False)
    # Descriptive follow-up after seeing how often the original graph was reproduced.
    simulations = rows[rows.condition.eq('input_mca')]
    groups = ['person','visit','half']
    distinct = simulations.groupby(groups).edge_sha256.nunique()
    changed = simulations[simulations.identical_offdiag.eq(0)]
    changed_cells = changed.groupby(groups).r2_loss.mean()
    assert len(changed_cells)==100
    changed_people = changed_cells.groupby(level='person').mean().to_numpy()
    sample = np.random.default_rng(20261119).integers(25,size=(10000,25))
    sensitivity = dict(changed_runs=len(changed),represented_cells=len(changed_cells),
                       mean_loss=changed_people.mean(),ci=np.quantile(changed_people[sample].mean(axis=1),[.025,.975]).tolist(),
                       unique_offdiag_per_cell_min=int(distinct.min()),unique_offdiag_per_cell_median=float(distinct.median()),
                       unique_offdiag_per_cell_max=int(distinct.max()),
                       unique_definition='among the 20 perturbation runs within each acquisition half, excluding the separate reference')
    (OUT/'changed_only.json').write_text(json.dumps(sensitivity,indent=2)+'\n',encoding='utf-8')
    meta=dict(archive_bytes=hdf.stat().st_size,archive_md5=HDF_MD5,source_url='https://zenodo.org/records/4041549',
              condition_order=KINDS,source_graphs=100,replicate_graphs=2000,participants=25,relative_penalty=PENALTY,
              fixed_input_map=10,relative_gain=transfer.GAIN,source_fit_steps=PILOT.FIT_N,test_streams=4,
              scored_steps_per_stream=512,external_library='vendored conn2res',score='mean of 10 delaywise R2',
              checked_reference_to_half_pairs=len(check),max_archive_error=archive_error,
              independent_target_checks=len(errors),independent_donors_per_graph=25,max_independent_error=max(errors),
              max_independent_similarity_error=max(correlations),identical_matrix_replays=int(rows.query("condition=='input_mca'").identical_offdiag.sum()),
              elapsed_seconds=perf_counter()-start)
    (OUT/'verification.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(summary.query("visit=='all' and metric in ['own_r2','top1','similarity','r2_loss']").round(4).to_string(index=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/nki/inputs.h5')
    main(p.parse_args().source)
