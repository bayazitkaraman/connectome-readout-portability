"""Audit zero-label transfer by similarity, acquisition, and source identity."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t

import nki_transfer as transfer
from paired_pool_augmentation import read_graphs


ROOT = transfer.ROOT
OUT = ROOT / "results/transfer-similarity"
CELL = ["src_visit","src_half","target_visit","target_half"]
KEY = CELL + ["target_id"]
BOUNDARIES = [-np.inf,.45,.60,.75,.85,.92,np.inf]
LABELS = ["<.45",".45–.60",".60–.75",".75–.85",".85–.92",">=.92"]


def fixed_nki(raw):
    """Replace the six exceptions to alpha=10 in the archived common-penalty sensitivity."""
    result = raw.copy()
    diff = raw[raw.alt_alpha!=10.]
    graphs,halves,_,ids,_ = transfer.load(transfer.DEFAULT_DATA)
    hi = dict(zip(halves, range(2)))
    si = dict(zip(ids, range(len(ids))))
    mapping = transfer.pilot.input_map(graphs.shape[-1],key=10)
    for idx,r in diff.iterrows():
        i = si[r.target_id]
        ident = 10000+(r.target_visit-1)*len(ids)+i
        net = transfer.pilot.reservoir(graphs[hi[r.target_half],r.target_visit-1,i],transfer.GAIN)
        test = transfer.pilot.combine([transfer.pilot.simulate(net,mapping,ident,61,512,k) for k in range(4)])
        for participant,label in [(r.target_id,"own"),(r.other_id,"other")]:
            s = si[participant]
            net = transfer.pilot.reservoir(graphs[hi[r.src_half],r.src_visit-1,s],transfer.GAIN)
            train = transfer.pilot.simulate(net,mapping,10000+(r.src_visit-1)*len(ids)+s,0,2352)
            model = transfer.pilot.RidgeFamily.from_design(train)
            score = lambda a: transfer.errors(model.predict(test[0],a),test)[1].mean()
            assert np.isclose(score(r.alt_alpha),r["alt_"+label+"_r2"],atol=1e-10,rtol=1e-10)
            result.loc[idx,"alt_"+label+"_r2"] = score(10.)
        result.loc[idx,"alt_alpha"] = 10.
    assert result.groupby(KEY).alt_own_r2.agg(lambda x:x.max()-x.min()).max()<1e-12
    return result


def beijing():
    f = pd.read_csv(ROOT/"results/density-controls/frozen/single_donor_pairs.csv",
                    dtype={"target_id":str,"other_id":str})
    f = f.query("cohort=='bnu1' and source_input=='original_distinct_inputs'").copy()
    assert f.relative_penalty.eq(1.).all()
    g,ids,_ = read_graphs("bnu1")
    a,b = np.triu_indices(g.shape[-1],1)
    e = g[:,:,a,b].astype(float)
    e -= e.mean(axis=2,keepdims=True)
    e /= np.linalg.norm(e,axis=2,keepdims=True)
    sim = pd.DataFrame(e[1]@e[0].T,index=ids,columns=ids)
    f["own_similarity"]=[sim.loc[x,x] for x in f.target_id]
    f["other_similarity"]=[sim.loc[x,y] for x,y in zip(f.target_id,f.other_id)]
    f["src_visit"],f["target_visit"]=1,2
    f["src_half"],f["target_half"]="all","all"
    f.rename(columns={"own_r2":"alt_own_r2","other_r2":"alt_other_r2"},inplace=True)
    return f


def long_table(pairs):
    # Each own scan readout contributes once per cell and target.
    own=pairs.groupby(KEY,as_index=False)[["own_similarity","alt_own_r2"]].mean()
    own["other_id"]=own.target_id
    own=own.rename(columns={"own_similarity":"similarity","alt_own_r2":"score"})
    other=pairs[KEY+["other_id","other_similarity","alt_other_r2"]].rename(
        columns={"other_similarity":"similarity","alt_other_r2":"score"})
    data=pd.concat([own,other],ignore_index=True)
    data["identity"]=(data.target_id==data.other_id).astype(int)
    data["same_acquisition"]=(data.identity.astype(bool) & (data.src_visit==data.target_visit)).astype(int)
    data["source_type"]=np.select([data.same_acquisition.astype(bool),data.identity.astype(bool)],
                                   ["own_same_acquisition","own_cross_visit"],default="stranger")
    data["bin"]=pd.cut(data.similarity,BOUNDARIES,labels=LABELS,right=False)
    return data


def bin_summary(frame):
    rows = []
    for (group,b),p in frame.groupby(["source_type","bin"],observed=True):
        rows.append({"source_type":group,"bin":b,"mean_score":p.score.mean(),"mean_similarity":p.similarity.mean(),
                     "n":len(p),"n_target_ids":p.target_id.nunique(),"n_source_ids":p.other_id.nunique()})
    return pd.DataFrame(rows)


def group_outer(scores,groups):
    grouped=scores.groupby(groups,sort=False).sum().to_numpy()
    return grouped.T @ grouped


def regression(frame,terms):
    cols=["intercept"]+terms
    x=np.column_stack([np.ones(len(frame)),frame[terms].to_numpy()])
    y=frame.score.to_numpy()
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=y-x@beta
    scores=pd.DataFrame(x*residual[:,None],index=frame.index)
    n,p=x.shape
    inv=np.linalg.inv(x.T@x)
    ids=np.unique(np.r_[frame.target_id.to_numpy(),frame.other_id.to_numpy()])
    # Target-cluster CR1 and small-cluster t reference.
    g=frame.target_id.nunique()
    factor=g/(g-1)*(n-1)/(n-p)
    target_meat=group_outer(scores,frame.target_id)*factor
    # Dyadic sandwich: count any two observations sharing a participant in either
    # role once, subtract duplicate contributions on non-self unordered pairs.
    cross=frame.target_id!=frame.other_id
    second=scores.loc[cross]
    endpoint_scores=pd.concat([scores,second],ignore_index=True)
    endpoint_ids=pd.concat([frame.target_id,frame.loc[cross,"other_id"]],ignore_index=True)
    unordered=pd.Series(["|".join(sorted((a,b))) for a,b in
                         zip(frame.loc[cross,"target_id"],frame.loc[cross,"other_id"])],index=second.index)
    dyadic_meat=group_outer(endpoint_scores,endpoint_ids)
    if cross.any():
        dyadic_meat-=group_outer(second,unordered)
    dyadic_meat*=len(ids)/(len(ids)-1)*(n-1)/(n-p)
    rows=[]
    for method,meat in [("target_cluster",target_meat),("any_shared_participant",dyadic_meat)]:
        cov=inv@meat@inv
        for k,term in enumerate(cols):
            var=cov[k,k]
            se=np.sqrt(var) if var>=0 else np.nan
            width=t.ppf(.975,g-1)*se
            rows.append({"cluster":method,"term":term,"estimate":beta[k],"se":se,
                         "lo":beta[k]-width,"hi":beta[k]+width,"n":n,"n_targets":g})
    return pd.DataFrame(rows)


def matches(frame):
    rows = []
    for key,part in frame.groupby(KEY):
        own=part[part.identity==1].iloc[0]
        donors=part[part.identity==0].sort_values(["similarity","other_id"])
        near=donors.iloc[np.argmin(abs(donors.similarity.to_numpy()-own.similarity))]
        base=dict(zip(KEY,key))
        base.update(source_type=own.source_type,own_similarity=own.similarity,own_score=own.score,
                    max_other=donors.similarity.max(),min_other=donors.similarity.min())
        for caliper in [.025,.05]:
            distance=abs(near.similarity-own.similarity)
            if distance<=caliper:
                rows.append(dict(**base,rule=f"nearest_{caliper}",foreign_score=near.score,
                                 gain=own.score-near.score,mismatch=distance))
        lower=donors[donors.similarity<=own.similarity]
        upper=donors[donors.similarity>=own.similarity]
        if len(lower) and len(upper):
            a,b=lower.iloc[-1],upper.iloc[0]
            other=np.interp(own.similarity,[a.similarity,b.similarity],[a.score,b.score])
            rows.append(dict(**base,rule="interpolation",foreign_score=other,
                             gain=own.score-other,mismatch=b.similarity-a.similarity))
    return pd.DataFrame(rows)


def match_summary(frame,m):
    rows=[]
    for group,part in frame[frame.identity==1].groupby("source_type"):
        for rule in ["interpolation","nearest_0.025","nearest_0.05"]:
            matched=m[(m.source_type==group)&(m.rule==rule)]
            # Balance participants among retained overlap cases, preserving all of
            # each person's correlated acquisition/half directions in resampling.
            per=matched.groupby("target_id").gain.mean()
            n=len(per)
            mean=lo=hi=np.nan
            if n:
                mean=per.mean()
                if n>=2:
                    ix=np.random.default_rng(20261120).integers(n,size=(10000,n))
                    lo,hi=np.quantile(per.to_numpy()[ix].mean(axis=1),[.025,.975])
            rows.append(dict(source_type=group,rule=rule,eligible_cells=len(part),matched_cells=len(matched),
                             eligible_people=part.target_id.nunique(),matched_people=n,
                             mean_gain=mean,lo=lo,hi=hi,median_gap=matched.mismatch.median()))
    return pd.DataFrame(rows)


def own_acquisition_checks(table):
    own=table.loc[table.identity==1].copy()
    terms=["similarity","same_acquisition"]
    # Resample all 12 observations of a person together. Accumulate cross-products
    # so each replicate fits the same OLS model as the reported point estimate.
    xtx,xty=[],[]
    for _,g in own.groupby("target_id"):
        x=np.column_stack([np.ones(len(g)),g[terms]])
        xtx.append(x.T@x)
        xty.append(x.T@g.score.to_numpy())
    rng=np.random.default_rng(20261120)
    n=len(xtx)
    weights=rng.multinomial(n,np.full(n,1/n),size=10000)
    gram=np.einsum("bj,jkl->bkl",weights,np.array(xtx))
    rhs=(weights@np.array(xty))[:,:,None]
    draws=np.linalg.solve(gram,rhs)[:,-1,0]
    lo,hi=np.quantile(draws,[.025,.975])
    result={"own_same_acquisition_whole_participant_bootstrap":[float(lo),float(hi)]}
    # Compare own same vs across on the exact same target, with the source half
    # held fixed. Only the source acquisition changes in each paired contrast.
    cols=["target_id","target_visit","target_half","src_half"]
    same=own[own.same_acquisition==1]
    both=same.merge(own[own.same_acquisition==0],on=cols,suffixes=("_same","_cross"),validate="one_to_one")
    gap=abs(both.similarity_same-both.similarity_cross)
    overlaps={}
    for tol in [.025,.05,.1]:
        matched=both[gap<=tol].copy()
        matched["diff"]=matched.score_same-matched.score_cross
        per=matched.groupby("target_id")["diff"].mean()
        overlaps[str(tol)]=dict(cells=len(matched),of_cells=len(both),people=len(per),
                                of_people=n,mean_gain=float(per.mean()) if len(per) else None)
    result["own_same_vs_cross_within_absolute_similarity_gap"]=overlaps
    return result


def interpolation_leave_one_person_out(table):
    """Delete each identity in both roles and rematch eligible cross-visit cells."""
    people=sorted(table.target_id.unique())
    contrasts={person:{} for person in people}
    within=table[table.target_visit!=table.src_visit]
    for key,part in within.groupby(KEY):
        target=key[-1]
        own=part[part.identity==1].iloc[0]
        donors=part[part.identity==0].sort_values(["similarity","other_id"])
        similarity=donors.similarity.to_numpy()
        score=donors.score.to_numpy()
        ids=donors.other_id.to_numpy()
        for excluded in people:
            if excluded==target:
                continue
            keep=ids!=excluded
            r=similarity[keep]
            if r[0]<=own.similarity<=r[-1]:
                diff=own.score-np.interp(own.similarity,r,score[keep])
                contrasts[excluded].setdefault(target,[]).append(diff)
    rows=[]
    for excluded,targets in contrasts.items():
        gains=[np.mean(x) for x in targets.values()]
        rows.append(dict(excluded_id=excluded,matched_people=len(gains),
                         matched_cells=sum(map(len,targets.values())),mean_gain=np.mean(gains)))
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(ROOT/"results/nki-split-half/pair_scores.csv")
    fixed=fixed_nki(original)
    changed=original[original.alt_alpha!=10]
    counts=changed.groupby(KEY).ngroups
    fn=long_table(fixed)
    ref=long_table(original)
    check={"n_changed_pair_alphas":len(changed),"n_affected_own_targets_conditions":counts,
           "nki_max_own_score_change_after_deduplication":float((ref.score-fn.score)[ref.identity==1].abs().max()),
           "nki_max_donor_score_change":float((ref.score-fn.score)[ref.identity==0].abs().max())}
    check.update(own_acquisition_checks(fn))
    for cohort,ordinary,table in (("nki",ref,fn),("bnu1",None,long_table(beijing()))):
        bins=bin_summary(table)
        bins.to_csv(OUT/f"similarity_bins_{cohort}.csv",index=False)
        if ordinary is not None:
            bin_summary(ordinary).to_csv(OUT/"similarity_bins_nki_archived.csv",index=False)
        reg=[]
        for name,f,terms in (("own",table[table.identity==1],["similarity","same_acquisition"]),
                             ("all",table,["similarity","identity","same_acquisition"])):
            if cohort=="bnu1":
                terms=[t for t in terms if t!="same_acquisition"]
            view=f.copy()
            reg.append(regression(view,terms).assign(model=name))
        if ordinary is not None:
            reg.append(regression(ordinary[ordinary.identity==1],["similarity","same_acquisition"])
                       .assign(model="own_archived_23_donor_tuning"))
        reg=pd.concat(reg,ignore_index=True)
        reg.to_csv(OUT/f"similarity_regression_{cohort}.csv",index=False)
        m=matches(table)
        m.to_csv(OUT/f"similarity_matched_cells_{cohort}.csv",index=False)
        match_summary(table,m).to_csv(OUT/f"similarity_matches_{cohort}.csv",index=False)
        sensitivity=interpolation_leave_one_person_out(table)
        sensitivity.to_csv(OUT/f"similarity_matches_leave_one_out_{cohort}.csv",index=False)
        check[cohort+"_matched_gain_leave_one_out_range"]=[sensitivity.mean_gain.min(),sensitivity.mean_gain.max()]
        r=table[table.identity==1].similarity
        check[cohort+"_own_similarity_quartiles"]=r.quantile([0,.25,.5,.75,1]).to_dict()
        if cohort=="bnu1":
            check["bnu1_own_count_084_088"]=int(r.between(.84,.88).sum())
            check["bnu1_own_count_085_092"]=int(r.between(.85,.92).sum())
    late=fn[(fn.identity==1)&(fn.src_half=="odd")&(fn.target_half=="even")]
    check["matched_target_even_from_odd"]={}
    for v,group in late.groupby("target_visit"):
        a=group[group.src_visit==v].set_index("target_id").score.sort_index()
        b=group[group.src_visit!=v].set_index("target_id").score.sort_index()
        check["matched_target_even_from_odd"][str(v)]=dict(same=float(a.mean()),different=float(b.mean()),
                                                            people_same_higher=int((a>b).sum()))
    (OUT/"similarity_penalty_checks.json").write_text(json.dumps(check,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
