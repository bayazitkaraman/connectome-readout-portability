"""Rebuild the scalar nonlinear endpoint with a separate recurrence and ridge solver."""
import json
from pathlib import Path

from verify_results import SingularRidge, fixtures, join, sampled, stream
from verify_nki_calibration import GRID, adjacency
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/nonlinear-recall"


def dataset(graph, inputs, ident, role, n, repeat=0):
    x,_=fixtures(graph,inputs,ident,role,n,repeat)
    u=stream(11,ident,role,repeat).uniform(-1,1,n+200)
    y=u[199:-1]*u[198:-2]*u[197:-3]
    return x,y[:,None]


def heldout(fit,a,test):
    x,y=test
    return 1-float(((fit.predict(x,a)-y)**2).mean()/y.var())


def audit(cohort,graphs,ids):
    n=len(ids)
    map_sites=stream(10).choice(graphs.shape[-1],8,replace=False)
    inputs=np.zeros(graphs.shape[-1])
    inputs[map_sites[:4]],inputs[map_sites[4:]]=.5,-.5
    pair=pd.read_csv(OUT/"pair_scores.csv",dtype={"target_id":str,"other_id":str}).query("cohort==@cohort")
    person=pd.read_csv(OUT/"participant_scores.csv",dtype={"target_id":str}).query("cohort==@cohort").set_index("target_id")
    pools=json.loads((OUT/"pool_allocations.json").read_text())[cohort]
    fits,training,validation,tune_loss=[],[],[],[]
    for i,g in enumerate(graphs[0]):
        gs=adjacency(g)
        gt=adjacency(graphs[1,i])
        train=dataset(gs,inputs,10000+i,0,2352)
        training.append(train)
        validation.append(dataset(gs,inputs,10000+i,1,1176))
        model=SingularRidge(train)
        fits.append(model)
        x,y=dataset(gt,inputs,10000+n+i,1,1176)
        tune_loss.append([np.mean((model.predict(x,a)-y)**2) for a in GRID])
    tune_loss=np.asarray(tune_loss)
    index={str(pid):i for i,pid in enumerate(ids)}
    selected={0,n//2,n-1}
    targets={}
    errors=[]
    for i in selected:
        ident=10000+n+i
        g=adjacency(graphs[1,i])
        test=join([dataset(g,inputs,ident,61,512,k) for k in range(4)])
        targets[i]=test
        local=SingularRidge(dataset(g,inputs,ident,0,2352))
        a=local.tune(dataset(g,inputs,ident,1,1176),GRID)
        p=person.loc[str(ids[i])]
        np.testing.assert_allclose(a,p.local_alpha,rtol=1e-12,atol=0)
        errors.append(abs(heldout(local,a,test)-p.local_r2))
        splits=pools[str(ids[i])]
        def subset(name,bank):
            return join([sampled(bank[index[s]],rows) for s,rows in splits[name].items()])
        inner=SingularRidge(subset("inner_fit",training))
        pooled_alpha=inner.tune(subset("inner_validation",validation),GRID)
        assert np.isclose(pooled_alpha,p.pooled_alpha,rtol=1e-12,atol=0.)
        pooled=SingularRidge(subset("final_fit",training))
        errors.append(abs(heldout(pooled,pooled_alpha,test)-p.pooled_r2))
    checked=0
    for r in pair.itertuples():
        t,d=index[r.target_id],index[r.other_id]
        mask=np.ones(n,bool)
        mask[[t,d]]=False
        best=GRID[tune_loss[mask].mean(axis=0).argmin()]
        np.testing.assert_allclose(best,r.alpha,atol=0.,rtol=1e-12)
        if t in selected:
            errors.append(abs(heldout(fits[t],best,targets[t])-r.own_r2))
            errors.append(abs(heldout(fits[d],best,targets[t])-r.other_r2))
            checked+=1
    assert len(pair)==n*(n-1) and max(errors)<1e-7
    return dict(people=n,selection_replays=len(pair),own_and_stranger_score_pairs_replayed=checked,
                local_and_pooled_examples=len(selected),max_absolute_r2_difference=max(errors))


if __name__=="__main__":
    output={}
    for cohort,file in [("nki","nki/nki_reference.npz"),("bnu1","replication_candidates/corr/bnu1_dkt_retest.npz")]:
        with np.load(ROOT/"data"/file,allow_pickle=False) as raw:
            graph=raw["weights"]
            if cohort=="nki":
                graph=graph[raw["halves"].tolist().index("odd")]
            output[cohort]=audit(cohort,graph,raw["subjects"].astype(str))
    (OUT/"independent_verification.json").write_text(json.dumps(output,indent=2)+"\n")
    print(output)
