"""Check the all-donor, single-scan comparison at exactly 32 labeled target steps."""
import os
for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from frozen_single_donors import target_data
from paired_pool_augmentation import read_graphs
import nki_transfer as transfer
import nki_fewshot as fit
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/nki-split-half"
N = 32


def calibration_streams(matrix, index, count, mapping):
    net = transfer.pilot.reservoir(matrix, transfer.GAIN)
    identity = 10000 + count + index
    return [[transfer.pilot.simulate(net, mapping, identity, 60, N//4, 4*r+fold)
             for fold in range(4)] for r in range(fit.REPEATS)]


def score_target(job):
    priors, streams, test = job
    keys = list(priors)
    frozen, adapted = {}, {}
    for begin in range(0,len(keys),3):
        batch = keys[begin:begin+3]
        padding = batch + [batch[-1]]*(3-len(batch))
        w = np.stack([priors[key][0] for key in padding],axis=1)
        b = np.stack([priors[key][1] for key in padding])
        var = test[1].var(axis=0)
        baseline_mse = np.mean(fit.residuals(*test,w,b)**2,axis=0)
        baseline_r2 = (1-baseline_mse/var).mean(axis=1)
        trained = []
        for calibration in streams:
            selected = fit.choose(calibration,w,b)
            model = fit.ResidualFit(*transfer.pilot.combine(calibration),w,b)
            mse = model.test_errors(*test,w,b,selected)
            trained.append((1-mse/var).mean(axis=1))
        for i,key in enumerate(batch):
            frozen[key] = baseline_r2[i]
            adapted[key] = [r[i] for r in trained]
    return frozen, adapted


def source_models(graphs, mapping):
    result = []
    for index, graph in enumerate(graphs[0]):
        net = transfer.pilot.reservoir(graph, transfer.GAIN)
        train = transfer.pilot.simulate(net,mapping,10000+index,0,transfer.pilot.FIT_N)
        result.append(transfer.pilot.RidgeFamily.from_design(train))
    return result


def run(cohort):
    graphs, ids, half = read_graphs(cohort)
    n = len(ids)
    mapping, _, tests = target_data(graphs)
    models = source_models(graphs,mapping)
    folder = ROOT/"results/density-controls/frozen"
    reference = pd.read_csv(folder/"single_donor_pairs.csv", dtype={"target_id":str,"other_id":str})
    reference = reference.query("cohort==@cohort and source_input=='original_distinct_inputs'")
    calibration = pd.read_csv(ROOT/f"results/{cohort}-calibration/scores.csv",dtype={"subject":str})
    scratch = calibration.query("half==@half and target_visit==2 and source_visit==1 and budget==32 and method=='scratch'")
    scratch = scratch.pivot(index="subject", columns="repeat", values="r2")
    lookup = {str(sub):i for i,sub in enumerate(ids)}
    def jobs():
        for i,name in enumerate(ids):
            rows = reference[reference.target_id==name]
            keys = {(name, row.relative_penalty) for row in rows.itertuples()}
            keys.update((row.other_id,row.relative_penalty) for row in rows.itertuples())
            priors = {}
            for donor,alpha in sorted(keys):
                model = models[lookup[donor]]
                weights = model.weights(alpha)
                priors[donor,alpha] = weights, model.my-model.mx@weights
            priors["scratch"] = np.zeros_like(weights),np.zeros_like(model.my)
            yield priors, calibration_streams(graphs[1,i], i, n, mapping), tests[i]
    rows = []
    frozen_gap, scratch_gap = 0., 0.
    with ProcessPoolExecutor(max_workers=5) as pool:
        for name,(frozen, adapted) in zip(ids,pool.map(score_target,jobs())):
            scratch_gap = max(scratch_gap,max(abs(np.array(adapted["scratch"])-scratch.loc[name].to_numpy())))
            for r in reference[reference.target_id==name].itertuples():
                own, other = (name,r.relative_penalty),(r.other_id,r.relative_penalty)
                frozen_gap=max(frozen_gap,abs(frozen[own]-r.own_r2),abs(frozen[other]-r.other_r2))
                for rep,(a,b) in enumerate(zip(adapted[own],adapted[other])):
                    rows.append({"cohort":cohort,"subject":name,"donor":r.other_id,"repeat":rep,
                                 "own_r2":a,"other_r2":b,"gain":a-b})
    assert frozen_gap < 1e-10 and scratch_gap < 1e-8, (frozen_gap,scratch_gap)
    df=pd.DataFrame(rows)
    assert len(df)==n*(n-1)*6
    people=df.groupby(["cohort","subject"])[["own_r2","other_r2","gain"]].mean().reset_index()
    gains=people.gain.to_numpy()
    picks=np.random.default_rng(20261120).multinomial(n,np.ones(n)/n,size=10000)
    result={"cohort":cohort,"n":n,"ordered_pairs":n*(n-1),"budget":N,
            "own_r2":float(people.own_r2.mean()),"other_r2":float(people.other_r2.mean()),
            "gain":float(gains.mean()),"ci":np.quantile(picks@gains/n,[.025,.975]).tolist(),
            "max_subject_mean_absolute_gain":float(abs(gains).max()),
            "max_frozen_discrepancy":float(frozen_gap),"max_scratch_discrepancy":float(scratch_gap)}
    return df, result


if __name__ == "__main__":
    all_rows, checks = [], []
    for dataset in ("nki","bnu1"):
        records, summary=run(dataset)
        all_rows.append(records)
        checks.append(summary)
        print(summary,flush=True)
    OUT.mkdir(exist_ok=True)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/"single_scan_32_scores.csv",index=False)
    (OUT/"single_scan_32_summary.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
