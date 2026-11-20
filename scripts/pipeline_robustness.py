"""One-factor repeats of the zero-label linear decoder experiment on matched participants."""
import json
from pathlib import Path

from paired_pool_augmentation import read_graphs, transfer
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

pilot = transfer.pilot
OUT = transfer.ROOT/"results/pipeline-robustness"
DATA = transfer.ROOT/"data/replication_candidates/corr"
BOUNDS = [-np.inf,.45,.60,.75,.85,.92,np.inf]
BIN_NAMES = ["<0.45","0.45–0.60","0.60–0.75","0.75–0.85","0.85–0.92","≥0.92"]


def threshold(graphs, count, n_people):
    upper = np.triu_indices(graphs.shape[-1],1)
    result = np.zeros_like(graphs)
    tie = np.random.default_rng(np.random.SeedSequence([61973,n_people])).permutation(len(upper[0]))
    for g,out in zip(graphs.reshape(-1,*graphs.shape[-2:]),result.reshape(-1,*graphs.shape[-2:])):
        weights = g[upper]
        chosen = np.lexsort((tie,-weights.astype(float)))[:count]
        assert weights[chosen].min()>0
        a,b=upper[0][chosen],upper[1][chosen]
        out[a,b]=out[b,a]=weights[chosen]
    return result


def r2_matrix(graphs, mapping, radius, penalties):
    n=graphs.shape[1]
    slopes={a:[] for a in penalties}
    offsets={a:[] for a in penalties}
    for s,g in enumerate(graphs[0]):
        net = pilot.reservoir(g,radius)
        fit=pilot.RidgeFamily(pilot.Moment.from_data(pilot.simulate(net,mapping,10000+s,0,pilot.FIT_N)))
        for a in penalties:
            w=fit.weights(a)
            slopes[a].append(w)
            offsets[a].append(fit.my-fit.mx@w)
    slopes={a:np.asarray(w) for a,w in slopes.items()}
    offsets={a:np.asarray(b) for a,b in offsets.items()}
    r2={a:np.zeros((n,n)) for a in penalties}
    for t,g in enumerate(graphs[1]):
        net=pilot.reservoir(g,radius)
        test=pilot.combine([pilot.simulate(net,mapping,10000+n+t,61,pilot.TEST_N,k) for k in range(pilot.TEST_STREAMS)])
        x,y=test
        xc=x-x.mean(axis=0)
        yc=y-y.mean(axis=0)
        xx=xc.T@xc/len(x)
        xy=xc.T@yc/len(x)
        vary=y.var(axis=0)
        for a in penalties:
            w=slopes[a]
            b=offsets[a]
            delta=np.einsum("d,ndk->nk",x.mean(axis=0),w)+b-y.mean(axis=0)
            wx=(xx@w.transpose(1,0,2).reshape(xx.shape[0],-1)).reshape(xx.shape[0],*delta.shape).transpose(1,0,2)
            mse=delta**2+(w*wx).sum(axis=1)-2*np.einsum("ndk,dk->nk",w,xy)+vary
            r2[a][t]=(1-mse/vary).mean(axis=1)
            if t==0:
                np.testing.assert_allclose(r2[a][0,0],(1-((x@w[0]+b[0]-y)**2).mean(axis=0)/vary).mean(),atol=1e-10)
    return r2


def scores(cohort,scenario,graphs,ids,mapping,radius,penalties,variants):
    u,v=np.triu_indices(graphs.shape[-1],1)
    e=graphs[:,:,u,v].astype(float)
    e-=e.mean(axis=-1,keepdims=True)
    norm=np.linalg.norm(e,axis=-1,keepdims=True)
    assert (norm>0).all()
    e/=norm
    sim=e[1]@e[0].T
    r2=r2_matrix(graphs,mapping,radius,penalties)
    n=len(ids)
    t,d=np.indices((n,n))
    frames=[]
    for pen in penalties:
        name=variants.get(pen,scenario)
        frames.append(pd.DataFrame(dict(cohort=cohort,scenario=name,target=ids[t.ravel()],source=ids[d.ravel()],
                                        own=(t==d).ravel(),radius=radius,relative_ridge=pen,
                                        similarity=sim.ravel(),r2=r2[pen].ravel())))
    return frames


def schaefer_cases():
    case={}
    for s in (200,300):
        with np.load(DATA/f"bnu1_schaefer{s}.npz",allow_pickle=False) as raw:
            case[s]=(raw["weights"],raw["subjects"].astype(str),raw["centroids"])
    base=pilot.input_map(200,10)[0].ravel()
    anchors=np.flatnonzero(base)
    left=case[200][2][anchors]
    right=case[300][2]
    distances=cdist(left,right)
    distances[left[:,0,None]*right[:,0]<0]=np.inf
    rows,cols=linear_sum_assignment(distances)
    assert np.array_equal(rows,np.arange(8))
    m300=np.zeros(300)
    m300[cols]=base[anchors]
    inputs=pd.DataFrame(dict(parcel_200=anchors+1,parcel_300=cols+1,input_weight=m300[cols],
                             centroid_distance_mm=distances[rows,cols]))
    inputs.to_csv(OUT/"scale_input_sites.csv",index=False)
    frames=[]
    for s,m in [(200,base),(300,m300)]:
        graphs,ids,_=case[s]
        frames+=scores("bnu1",f"schaefer_{s}",graphs,ids,(m.reshape(1,-1),np.flatnonzero(m==0)),.9,[1.],{})
    return frames


def summarize(pairs):
    people=[]
    for (cohort,scenario,target),group in pairs.groupby(["cohort","scenario","target"],sort=False):
        own=group[group.own].iloc[0]
        others=group[~group.own]
        closest=others[others.similarity.eq(others.similarity.max())]
        people.append(dict(cohort=cohort,scenario=scenario,target=target,
                           nearest_r2=closest.r2.mean(),mean_other_r2=others.r2.mean(),own_r2=own.r2,
                           gain=closest.r2.mean()-others.r2.mean(),nearest_similarity=closest.similarity.mean(),
                           top1=int(own.similarity>others.similarity.max()+1e-12)))
    people=pd.DataFrame(people)
    people.to_csv(OUT/"participant_scores.csv",index=False)
    rows=[]
    for (cohort,scenario),group in people.groupby(["cohort","scenario"],sort=False):
        g=group.gain.to_numpy()
        idx=np.random.default_rng(20261120).integers(len(g),size=(10000,len(g)))
        lo,hi=np.quantile(g[idx].mean(axis=1),[.025,.975])
        rows.append(dict(cohort=cohort,scenario=scenario,n=len(g),mean_other=group.mean_other_r2.mean(),
                         closest=group.nearest_r2.mean(),own=group.own_r2.mean(),
                         gain=g.mean(),gain_lower=lo,gain_upper=hi,people_improved=int((g>0).sum()),
                         wiring_top1=int(group.top1.sum())))
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)
    pairs["bin"]=pd.cut(pairs.similarity,BOUNDS,labels=BIN_NAMES,right=False)
    curves=pairs.groupby(["cohort","scenario","own","bin"],observed=True).agg(
        pairs=("r2","size"),people=("target","nunique"),mean_similarity=("similarity","mean"),mean_r2=("r2","mean"),
        fraction_r2_at_least_010=("r2",lambda s:(s>=.10).mean()),
        fraction_r2_at_least_050=("r2",lambda s:(s>=.50).mean())).reset_index()
    curves.to_csv(OUT/"similarity_curves.csv",index=False)


def main():
    frames=[]
    for cohort,pen,k in [("nki",10.,177),("bnu1",1.,1291)]:
        g,ids,_=read_graphs(cohort)
        native=pilot.input_map(g.shape[-1],10)
        transforms=[("native",g),("log1p",np.log1p(g.astype(float))),("binary",(g>0).astype(float)),
                    ("equal_edges",threshold(g,k,len(ids)))]
        for name,graphs in transforms:
            chosen=[pen*.1,pen,pen*10] if name=="native" else [pen]
            frames+=scores(cohort,name,graphs,ids,native,.9,chosen,{pen*.1:"ridge_x0.1",pen*10:"ridge_x10"})
        for radius in (.7,1.):
            frames+=scores(cohort,f"radius_{radius:.2f}",g,ids,native,radius,[pen],{})
        print(f"Finished {cohort} native-atlas controls",flush=True)
    frames+=schaefer_cases()
    pairs=pd.concat(frames,ignore_index=True)
    archived=pd.read_csv(transfer.ROOT/"results/density-controls/frozen/single_donor_pairs.csv",
                         dtype={"target_id":str,"other_id":str})
    original=archived[archived.source_input=="original_distinct_inputs"]
    checked=pairs.query("scenario=='native' and not own").merge(original,left_on=["cohort","target","source"],
                    right_on=["cohort","target_id","other_id"],validate="one_to_one")
    assert len(checked)==3792
    diff=float(abs(checked.other_r2-checked.r2).max())
    assert diff<1e-9
    pairs.to_csv(OUT/"pair_scores.csv",index=False)
    summarize(pairs)
    (OUT/"settings.json").write_text(json.dumps(dict(simulation_seed=pilot.SEED,input_map_key=10,
        train_role=0,test_role=61,train_stream="10000+source_index",test_stream="10000+n+target_index",
        test_repeats=[0,1,2,3],epochs_fit=2352,test_rows_per_stream=512,warmup=200,delay_range=[1,10],
        archived_native_stranger_scores_checked=len(checked),max_abs_r2_difference=diff),indent=2)+"\n")


if __name__=="__main__":
    main()
