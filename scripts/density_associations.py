"""Quantify edge counts and their contribution to matched scan/recall associations."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from analyze_anatomy_advantage import INPUTS, PAIR, PROJECT, RULES, rank_correlations


OUT = PROJECT / "results/density-controls/observational"
BUDGETS = (0, 8, 16, 32, 64, 128, 256)
SEED = 184021
NBOOT = 4000
TERMS = {
    "raw": ["raw"],
    "raw_plus_density": ["raw", "density"],
    "density_only": ["density"],
    "split_plus_density": ["presence", "shared_weight", "density"],
}


def pair_measures(target, first):
    shared = (target > 0) & (first > 0)
    both = np.logical_or(target > 0, first > 0)
    if shared.sum() < 3 or np.ptp(target[shared]) == 0 or np.ptp(first[shared]) == 0:
        raise ValueError("Insufficient shared-edge variation for Pearson correlation.")
    return dict(raw=np.corrcoef(target, first)[0, 1],
                presence=shared.sum()/both.sum(), shared_weight=np.corrcoef(target[shared],first[shared])[0,1],
                common_edges=int(shared.sum()))


def graphs_and_design(cohort):
    folder = PROJECT / "results" / f"{cohort}-augmentation"
    meta = json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
    with np.load(INPUTS[cohort], allow_pickle=False) as src:
        m = src["weights"]
        if cohort == "nki":
            m = m[src["halves"].astype(str).tolist().index("odd")]
        ids = src["subjects"].astype(str).tolist()
        assert ids == meta["subjects"] and m.shape == (2,len(ids),83,83)
        r, c = np.triu_indices(83, k=1)
        edges = m[:,:,r,c].astype(float)
    n_edges = edges.shape[-1]
    assert n_edges == 3403
    edge_counts = (edges > 0).sum(axis=-1)
    scans = [dict(cohort=cohort,subject=person,visit=v+1,nonzero_edges=int(edge_counts[v,i]),
                  possible_edges=n_edges,density=edge_counts[v,i]/n_edges)
             for i, person in enumerate(ids) for v in range(2)]
    pair_rows = []
    for entry in meta["records"]:
        target, other = ids.index(entry["subject"]), ids.index(entry["other_subject"])
        target_edges = edges[1,target]
        own_metrics = pair_measures(target_edges, edges[0,target])
        other_metrics = pair_measures(target_edges, edges[0,other])
        n_own, n_other, n_target = edge_counts[0,target], edge_counts[0,other], edge_counts[1,target]
        row = dict(cohort=cohort, **{k:entry[k] for k in PAIR},
                   n_own_first=int(n_own),n_other_first=int(n_other),n_target_second=int(n_target),
                   density=(abs(n_target-n_other)-abs(n_target-n_own))/n_edges)
        for k in own_metrics:
            row[f"own_{k}"] = own_metrics[k]
            row[f"other_{k}"] = other_metrics[k]
            if k != "common_edges":
                row[k] = own_metrics[k] - other_metrics[k]
        pair_rows.append(row)
    pair_data = pd.DataFrame(pair_rows)
    scores = pd.read_csv(folder/"scores.csv", dtype={"subject":str,"other_subject":str})
    scores = scores[scores.budget.isin(BUDGETS)]
    assert np.all(scores.groupby(PAIR+["budget","mix_rule","method"]).size() == 6)
    wide = scores.groupby(PAIR+["budget","mix_rule","method"]).r2.mean().unstack().reset_index()
    wide["gain"] = wide.pool_plus_own - wide.pool_plus_other
    whole = wide.merge(pair_data,on=PAIR,validate="many_to_one")
    assert len(whole) == len(ids)*3*len(BUDGETS)*len(RULES)
    previous = pd.read_csv(PROJECT/"results/anatomy-advantage/pairs.csv", dtype={"subject":str, "other_subject":str})
    compare = whole.query("budget == 8").merge(previous[previous.cohort==cohort],on=PAIR+["mix_rule"],validate="one_to_one")
    np.testing.assert_allclose(compare.raw, compare.anatomy_advantage, atol=1e-12)
    np.testing.assert_allclose(compare.gain, compare.accuracy_advantage, atol=1e-12)
    return whole, scans, edge_counts


def summarize_counts(cohort, counts):
    n = counts.shape[1]
    sets = []
    for name, data in (("visit_1", counts[0]),("visit_2", counts[1]),("both_visits",counts.ravel())):
        sets.append(dict(cohort=cohort,sample=name,n=len(data),mean=data.mean(),sd=data.std(ddof=1),
                         min=data.min(),p25=np.quantile(data,.25),median=np.median(data),
                         p75=np.quantile(data,.75),max=data.max()))
    def icc(a):
        grand = a.mean()
        targets, visits = a.mean(axis=0), a.mean(axis=1)
        ms_target = 2 * np.var(targets, ddof=1)
        ms_visit = a.shape[1] * np.var(visits,ddof=1)
        residual = a - targets[None,:] - visits[:,None] + grand
        ms_error = (residual**2).sum()/(a.shape[1]-1)
        return (ms_target-ms_error)/(ms_target+ms_error+2*(ms_visit-ms_error)/a.shape[1])
    diff = counts[1]-counts[0]
    draws = np.random.default_rng(SEED).integers(n,size=(NBOOT,n))
    icc_ci = np.quantile([icc(counts[:,draw]) for draw in draws], [.025,.975])
    correlation = stats.pearsonr(counts[0],counts[1])
    return sets, dict(cohort=cohort,n=n,pearson=correlation.statistic,pearson_ci_low=correlation.confidence_interval().low,
                      pearson_ci_high=correlation.confidence_interval().high,icc_absolute_single=icc(counts),
                      icc_ci_low=icc_ci[0],icc_ci_high=icc_ci[1],mean_signed_change=diff.mean(),
                      mean_abs_change=np.abs(diff).mean(),median_abs_change=np.median(abs(diff)),
                      median_abs_density_change=np.median(abs(diff))/3403)


def reml(y, z, target, donor):
    """Profile Gaussian restricted likelihood over two nonnegative variance ratios."""
    n, p = z.shape
    identity = np.eye(n)
    nt = np.equal.outer(target,target).astype(float)
    nd = np.equal.outer(donor,donor).astype(float)
    yz = np.column_stack([z,y])

    def details(ratios):
        cov = identity + ratios[0]*nt + ratios[1]*nd
        chol = linalg.cholesky(cov,lower=True,check_finite=False)
        white = linalg.solve_triangular(chol,yz,lower=True,check_finite=False)
        design, values = white[:,:-1], white[:,-1]
        xtx = design.T @ design
        beta = np.linalg.solve(xtx,design.T @ values)
        resid = values-design@beta
        noise = resid@resid/(n-p)
        objective = (n-p)*np.log(noise) + 2*np.log(np.diag(chol)).sum()+np.linalg.slogdet(xtx)[1]
        return objective, beta, noise*np.linalg.inv(xtx), noise

    bounds = ((0,10000),(0,10000))
    fitted = [optimize.minimize(lambda q:details(q)[0], start,method="L-BFGS-B", bounds=bounds,
                                options={"ftol":1e-12,"maxiter":200})
              for start in ([0,0],[1,1],[4,.1])]
    best = min(fitted,key=lambda x:x.fun)
    obj, beta, cov, noise = details(best.x)
    return beta, np.sqrt(np.diag(cov)), dict(converged=bool(best.success),objective=obj,target_var=noise*best.x[0],
            donor_var=noise*best.x[1],residual_var=noise,target_on_boundary=best.x[0]<1e-6,
            donor_on_boundary=best.x[1]<1e-6,optimizer_message=best.message)


def boot_coefficients(z,y,weights):
    n, k = z.shape
    gram = (weights @ (z[:,:,None]*z[:,None,:]).reshape(n,k*k)).reshape(-1,k,k)
    rhs = weights@(z*y[:,None])
    return (np.linalg.pinv(gram,hermitian=True) @ rhs[:,:,None])[:,:,0]


def boot_crossed_weights(frame, rng):
    t = pd.Categorical(frame.subject).codes
    d = pd.Categorical(frame.other_subject).codes
    nt, nd = t.max()+1,d.max()+1
    wc = rng.multinomial(nt,np.ones(nt)/nt,size=NBOOT)
    wd = rng.multinomial(nd,np.ones(nd)/nd,size=NBOOT)
    return wc[:,t]*wd[:,d]


def mixed_and_boot(data, cohort, rule, rng):
    data = data[(data.budget == 8)&(data.mix_rule == rule)]
    weights = boot_crossed_weights(data, rng)
    rows, drawsets = [],{}
    for model, terms in TERMS.items():
        y = data.gain.to_numpy()
        predictors = data[terms].to_numpy()
        standard = predictors.std(axis=0)
        z = np.column_stack([np.ones(len(data)), (predictors-predictors.mean(axis=0))/standard])
        beta, se, status = reml(y, z, data.subject.to_numpy(),data.other_subject.to_numpy())
        ols = np.linalg.lstsq(z,y,rcond=None)[0][1:] / standard
        draws = boot_coefficients(z,y,weights)[:,1:] / standard
        tcrit = stats.t.ppf(.975,data.subject.nunique()-1)
        for j,term in enumerate(terms):
            slope, stderr = beta[j+1]/standard[j], se[j+1]/standard[j]
            lo, hi = np.quantile(draws[:,j],[.025,.975])
            rows.append(dict(cohort=cohort,mix_rule=rule,model=model,term=term,n_pairs=len(data),
                             mixed_slope=slope, mixed_ci_low=slope-tcrit*stderr,mixed_ci_high=slope+tcrit*stderr,
                             ols_slope=ols[j],crossed_boot_ols_low=lo,crossed_boot_ols_high=hi,
                             design_condition_number=np.linalg.cond(z),**status))
            drawsets[(model,term)] = (ols[j],draws[:,j],slope,stderr)
    return rows, drawsets


def partial_rank(x,y,d):
    ranks = [stats.rankdata(arr,axis=-1) for arr in (x,y,d)]
    a,b,c = [arr-arr.mean(axis=-1,keepdims=True) for arr in ranks]
    cnorm = (c*c).sum(axis=-1,keepdims=True)
    a = a-((a*c).sum(axis=-1,keepdims=True)/cnorm)*c
    b = b-((b*c).sum(axis=-1,keepdims=True)/cnorm)*c
    return (a*b).sum(axis=-1)/np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))


def participant_analyses(data,cohort):
    groups = ["mix_rule","budget","subject"]
    cols = ["raw","density","gain","pool_plus_own","pool_plus_other","pool_only"]
    people = data.groupby(groups,as_index=False)[cols].mean()
    summaries,partial = [],[]
    n = people.subject.nunique()
    draws = np.random.default_rng(SEED).integers(n,size=(NBOOT,n))
    for rule, samples in people.groupby("mix_rule"):
        early = samples[samples.budget <= 32]
        pivot = early.pivot(index="subject",columns="budget",values="gain").sort_index()
        values = pivot.to_numpy() @ np.array([4,8,12,8])/32.
        auc = samples[samples.budget==0].sort_values("subject").copy()
        auc["gain"] = values
        for name in ["pool_plus_own","pool_plus_other","pool_only"]:
            v = early.pivot(index="subject",columns="budget",values=name).sort_index()
            auc[name] = v.to_numpy() @ np.array([4,8,12,8])/32.
        for budget,g in [*samples.groupby("budget"),("nAUC_0_32",auc)]:
            g = g.sort_values("subject")
            x,y = g.raw.to_numpy(),g.gain.to_numpy()
            rho = rank_correlations(x,y)
            rr = rank_correlations(x[draws],y[draws])
            rho_ci=np.nanquantile(rr,[.025,.975])
            centers = (x-x.mean())
            slope=centers@(y-y.mean())/(centers@centers)
            gain_ci=np.quantile(y[draws].mean(axis=1),[.025,.975])
            summaries.append(dict(cohort=cohort,mix_rule=rule,budget=budget,subjects=len(g),
                                  mean_own=g.pool_plus_own.mean(),mean_other=g.pool_plus_other.mean(),
                                  mean_pool_only=g.pool_only.mean(), mean_gain=y.mean(),gain_ci_low=gain_ci[0],
                                  gain_ci_high=gain_ci[1],target_gain_sd=y.std(ddof=1),max_abs_target_gain=max(abs(y)),
                                  spearman=rho,rho_ci_low=rho_ci[0],rho_ci_high=rho_ci[1],raw_slope=slope))
            if budget==8:
                d=g.density.to_numpy()
                sample=partial_rank(x,y,d)
                ci=np.nanquantile(partial_rank(x[draws],y[draws],d[draws]),[.025,.975])
                partial.append(dict(cohort=cohort,mix_rule=rule,subjects=len(x),partial_rho=sample,
                                    partial_ci_low=ci[0],partial_ci_high=ci[1],raw_rho=rho,
                                    rho_anatomy_density=stats.spearmanr(x,d).statistic,
                                    rho_gain_density=stats.spearmanr(y,d).statistic))
    return summaries,partial,people.query("budget==8")


def common_support(people):
    ranges=[]
    for rule,grp in people.groupby("mix_rule"):
        ref=grp[grp.cohort=="bnu1"].raw
        for name,q in (("beijing_full",(0,1)),("beijing_central_90pct",(.05,.95))):
            lo,hi=ref.quantile(list(q))
            for cohort,g in grp.groupby("cohort"):
                s=g[g.raw.between(lo,hi)]
                rho=s.raw.corr(s.gain,method="spearman") if len(s)>=3 else np.nan
                slope=stats.linregress(s.raw,s.gain).slope if len(s)>=3 else np.nan
                ranges.append(dict(mix_rule=rule,range_selection=name,cohort=cohort,min_beijing=lo,max_beijing=hi,
                                   n=len(s),total=len(g),spearman=rho,slope=slope))
    return ranges


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    scans,descriptions,agreement,models,budgets,partials,people,frames=[],[],[],[],[],[],[],[]
    all_draws={}
    for cohort in INPUTS:
        frame,s,c=graphs_and_design(cohort)
        frames.append(frame)
        scans.extend(s)
        sets,agr=summarize_counts(cohort,c)
        descriptions.extend(sets)
        agreement.append(agr)
        for rule in RULES:
            rows,sims=mixed_and_boot(frame,cohort,rule,np.random.default_rng(SEED+int(cohort=="bnu1")))
            models.extend(rows)
            all_draws[(cohort,rule)]=sims
        rows,partial,p=participant_analyses(frame,cohort)
        budgets.extend(rows)
        partials.extend(partial)
        people.append(p.assign(cohort=cohort))
    diff=[]
    for rule in RULES:
        nk,bj=all_draws[("nki",rule)],all_draws[("bnu1",rule)]
        for key in nk:
            a,b=nk[key],bj[key]
            lo,hi=np.quantile(a[1]-b[1],[.025,.975])
            delta,se=a[2]-b[2],np.hypot(a[3],b[3])
            # Conservative t reference uses the smaller cohort's n-1 degrees of freedom.
            q=stats.t.ppf(.975,24)
            diff.append(dict(mix_rule=rule,model=key[0],term=key[1],
                             nki_minus_beijing_mixed_slope=delta,mixed_ci_low=delta-q*se,mixed_ci_high=delta+q*se,
                             nki_minus_beijing_marginal_slope=a[0]-b[0],crossed_boot_ols_low=lo,crossed_boot_ols_high=hi))
    files={"scan_counts":scans,"count_distributions":descriptions,"test_retest":agreement,"crossed_models":models,
           "budget_curves":budgets,"target_partial_rank":partials,"slope_difference":diff,
           "restricted_range":common_support(pd.concat(people,ignore_index=True))}
    for filename,rows in files.items():
        pd.DataFrame(rows).to_csv(OUT/f"{filename}.csv",index=False)
    all_pairs = pd.concat(frames,ignore_index=True)
    all_pairs.to_csv(OUT/"matched_pairs_by_budget.csv",index=False)
    features = all_pairs[(all_pairs.budget == 8) & (all_pairs.mix_rule == "equal_donor")]
    features.groupby("cohort")[["raw","presence","shared_weight","density"]].corr(
        method="spearman").rename_axis(index=["cohort","feature"]).to_csv(OUT/"predictor_correlations.csv")
    print("Wrote scan counts, closed-form structural checks and crossed models for both cohorts.")


if __name__=="__main__":
    run()
