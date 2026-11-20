"""Compute repeatability of separately fitted, source-locally tuned scores in both native cohorts."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from . import analysis
from .metrics import reliability

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pipeline-robustness"
BOOTSTRAP_SEED = 20261120


def intervals(values):
    n=len(values)
    sample=np.random.default_rng(BOOTSTRAP_SEED).integers(n,size=(10000,n))
    x=values[sample]
    p=x.mean(axis=2,keepdims=True)
    v=x.mean(axis=1,keepdims=True)
    b=2*p[:,:,0].var(axis=1,ddof=1)
    m=n*v[:,0].var(axis=1,ddof=1)
    e=np.square(x-p-v+x.mean(axis=(1,2),keepdims=True)).sum(axis=(1,2))/(n-1)
    icc=(b-e)/(b+e+2*(m-e)/n)
    ranks=rankdata(x,axis=1)
    ranks-=ranks.mean(axis=1,keepdims=True)
    rho=(ranks[:,:,0]*ranks[:,:,1]).sum(axis=1)/np.sqrt(np.square(ranks).sum(axis=1).prod(axis=1))
    return {"icc_low":np.quantile(icc,.025),"icc_high":np.quantile(icc,.975),
            "rho_low":np.quantile(rho,.025),"rho_high":np.quantile(rho,.975)}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    inputs=(("nki",ROOT/"data/nki/nki_reference.npz","odd"),
            ("bnu1",ROOT/"data/replication_candidates/corr/bnu1_dkt_retest.npz",None))
    with (ROOT/"results/nonlinear-recall/participant_scores.csv").open(encoding="utf-8",newline="") as f:
        product_ref={(r["cohort"],r["target_id"]):float(r["local_r2"]) for r in csv.DictReader(f)}
    people,summary=[],[]
    with threadpool_limits(limits=1):
        for cohort,path,half in inputs:
            graphs,subjects,_,regions=analysis.load_data(path,half)
            n=len(subjects)
            seed=20261119
            mapping=analysis.make_map(seed,10,len(regions),8)
            for task,lags in (("recall",list(range(1,11))),("product",[1,2,3])):
                cfg=SimpleNamespace(task=task,lags=lags,warmup=200,penalties=analysis.DEFAULT_GRID,
                                    train_steps=2352,validation_steps=1176,test_steps=512,test_streams=4)
                scores=np.empty((n,2))
                error=0.
                for v,group in enumerate(graphs):
                    for i,g in enumerate(group):
                        scan=analysis.fit_scan(analysis.pilot.reservoir(g,.90),mapping,cfg,seed,10000+v*n+i)
                        scores[i,v]=scan.test.r2(scan.weights[[scan.local]]).mean()
                        if task=="product" and v==1:
                            error=max(error,abs(scores[i,v]-product_ref[cohort,subjects[i]]))
                        people.append(dict(cohort=cohort,task=task,subject=subjects[i],visit=v+1,
                                           local_alpha=cfg.penalties[scan.local],local_r2=scores[i,v]))
                if error>1.e-8:
                    raise AssertionError(f"{cohort}: archived target-fitted product score mismatch {error}")
                summary.append(dict(cohort=cohort,task=task,n=n,visit1_mean=scores[:,0].mean(),
                    visit2_mean=scores[:,1].mean(),**reliability(scores[:,0],scores[:,1]),
                    **intervals(scores),product_visit2_archive_max_abs_error=error if task=="product" else ""))
    analysis.write_rows(OUT/"fitted_score_reliability.csv",summary)
    analysis.write_rows(OUT/"fitted_score_reliability_people.csv",people)
    meta=dict(estimand="Repeatability of performance when the readout is refitted and tuned separately on each scan.",
        inputs="NKI odd-direction DKT, BNU1 whole-scan DKT; first and second acquisition. No study participants excluded.",
        targets="Per-scan mean of 10 single-lag R² (recall), or one three-lag product R².",
        seeds=dict(simulation=seed,map_key=10,bootstrap=BOOTSTRAP_SEED),
        streams="Independent by visit and person; key 10000 + visit_index*n + person_index. Roles 0/1/61 for training/validation/test. Four independent 512-step test streams.",
        fit_train_steps=2352,validation_steps=1176,warmup=200,spectral_radius=.9,
        penalties=analysis.DEFAULT_GRID.tolist(),selection="Independent source-local validation MSE minimization at each visit.",
        spearman="Pearson correlation of tied midranks across people between visits.",
        icc="Two-way absolute agreement single measurement ICC(A,1): (MS_person-MS_error)/(MS_person+MS_error+2*(MS_visit-MS_error)/n).",
        intervals="95% percentile intervals from 10,000 bootstrap draws of paired participants, conditional on simulation and test streams.",
        verification="Each second-visit product local R² checked against nonlinear-recall/participant_scores.csv.")
    (OUT/"fitted_score_reliability_settings.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
