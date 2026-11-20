"""Evaluate a source-only readout with a penalty learned on other people's repeat scans."""
import nki_transfer as fit
from analyze_nki_retest import analyze
import numpy as np
import pandas as pd


def run():
    path = fit.OUT
    out = path/"transfer-tuned"
    baseline = pd.read_csv(path/"summary.csv")
    main = baseline[(baseline.half=="odd") & (baseline.source_visit==1)].iloc[0]
    if main.median_self_r2 >= 0 and main.n_better_than_pooled >= main.people/2:
        print("The prospective condition for the sensitivity analysis did not occur.")
        return
    table = pd.read_csv(path/"pair_scores.csv",dtype={"source_subject":str, "target_subject":str})
    pooled = pd.read_csv(path/"pooled_scores.csv",dtype={"target_subject":str})
    graphs, halves, sessions, subjects, codes = fit.load(fit.DEFAULT_DATA)
    n = len(subjects)
    mapping = fit.pilot.input_map(graphs.shape[-1], key=10)
    results, summaries, persons, tuning = [], [], [], []
    for half, h in zip(halves, graphs):
        visits = [[fit.fit_scan(g,mapping,i,v,n) for i,g in enumerate(visit)] for v,visit in enumerate(h)]
        for sv,tv in [(0,1), (1,0)]:
            context = dict(half=half,source_visit=sv+1,target_visit=tv+1)
            def select(df):
                return df[(df.half==half)&(df.source_visit==sv+1)&(df.target_visit==tv+1)].copy()
            sources, targets = visits[sv],visits[tv]
            validation = np.empty((n,len(fit.PENALTIES)))
            for i,(source,target) in enumerate(zip(sources, targets)):
                for ai,alpha in enumerate(fit.PENALTIES):
                    pred = source.model.predict(target.validation[0],alpha)
                    validation[i,ai] = fit.errors(pred,target.validation)[0].mean()
                tuning.extend([dict(**context,subject=subjects[i],relative_ridge=a,validation_nmse=v)
                               for a,v in zip(fit.PENALTIES,validation[i])])
            scores=[]
            for j,target in enumerate(targets):
                for i,source in enumerate(sources):
                    keep = (np.arange(n)!=i)&(np.arange(n)!=j)
                    alpha = fit.PENALTIES[np.argmin(validation[keep].mean(axis=0))]
                    pred = source.model.predict(target.testing[0], alpha)
                    error,r2 = fit.errors(pred,target.testing)
                    scores.extend([dict(source_subject=subjects[i],target_subject=subjects[j],
                                        delay=k,transfer_alpha=alpha,normalized_mse=e,r2=r)
                                   for k,e,r in zip(fit.pilot.DELAYS,error,r2)])
            merged = select(table).drop(columns=["normalized_mse","r2"]).merge(pd.DataFrame(scores),
                    on=["source_subject","target_subject","delay"],validate="one_to_one")
            ps, sm = analyze(merged,select(pooled),primary_inference=False)
            summaries.append({**context,**sm})
            persons.append(ps.assign(**context))
            results.append(merged)
    out.mkdir(exist_ok=True)
    pd.concat(results).to_csv(out/"pair_scores.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
    pd.concat(persons).to_csv(out/"participant_summary.csv",index=False)
    pd.DataFrame(tuning).to_csv(out/"donor_validation.csv",index=False)


if __name__=="__main__":
    run()
