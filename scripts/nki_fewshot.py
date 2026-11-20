"""Estimate target-scan calibration curves with matched labeled time steps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import nki_transfer as previous
import numpy as np
import pandas as pd


pilot = previous.pilot
OUT = previous.ROOT / "results" / "nki-calibration"
BUDGETS = (0, 8, 16, 32, 64, 128, 256, 512, 2352)
REPEATS = 6
FOLDS = 4
METHODS = ("personal", "pooled", "scratch")
ALPHAS = previous.PENALTIES[::-1]
D = len(pilot.DELAYS)


def residuals(x, y, w0, b0):
    prediction = (x @ w0.reshape(len(w0), -1)).reshape(len(x), len(METHODS), D)
    return y[:, None, :] - prediction - b0


class ResidualFit:
    """One centered design SVD for three residual regressions and all penalties."""

    def __init__(self, x, y, w0, b0):
        self.mx = x.mean(axis=0)
        residual = residuals(x, y, w0, b0)
        self.mr = residual.mean(axis=0)
        u, s, vt = np.linalg.svd(x - self.mx, full_matrices=False)
        self.v = vt.T
        scaled_penalty = (s @ s) / x.shape[1] * ALPHAS
        projected = u.T @ (residual-self.mr).reshape(len(x), -1)
        # Scale coordinates before forming validation-error moments. This keeps
        # very small singular directions numerically manageable at weak ridge.
        self.coordinate_scale = np.maximum(s, np.sqrt(scaled_penalty[-1]))
        self.coordinate_scale[self.coordinate_scale == 0] = 1
        denom = s[:,None]**2 + scaled_penalty
        factors = np.divide(s[:,None] * self.coordinate_scale[:,None], denom,
                            out=np.zeros_like(denom), where=denom>0)
        self.coef = projected[:,None,:] * factors[:,:,None]

    def coordinates(self, x):
        return ((x-self.mx) @ self.v) / self.coordinate_scale

    def validation_mse(self, x, y, w0, b0):
        err_keep = residuals(x,y,w0,b0)
        keep = np.mean(err_keep**2, axis=(0,2))
        base = (err_keep-self.mr).reshape(len(x),-1)
        coords = self.coordinates(x)
        cross = coords.T @ base
        gram = coords.T @ coords
        flat = self.coef.reshape(len(gram),-1)
        quadratic = (gram @ flat).reshape(self.coef.shape)
        ss = (base*base).sum(axis=0) + (self.coef*(quadratic-2*cross[:,None,:])).sum(axis=0)
        return np.vstack([keep, ss.reshape(len(ALPHAS),len(METHODS),D).mean(axis=2)/len(x)])

    def test_errors(self, x, y, w0, b0, selections):
        err = residuals(x,y,w0,b0)
        coords = self.coordinates(x)
        for m,k in enumerate(selections):
            if k:
                coeff = self.coef[:,k-1,m*D:(m+1)*D]
                err[:,m] -= self.mr[m] + coords @ coeff
        return (err**2).mean(axis=0)


def choose(four, w0, b0):
    fold_scores = []
    for index, valid in enumerate(four):
        x,y = pilot.combine(four[:index]+four[index+1:])
        model = ResidualFit(x,y,w0,b0)
        fold_scores.append(model.validation_mse(*valid,w0,b0))
    losses = np.mean(fold_scores,axis=0)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite out-of-stream validation errors.")
    return np.argmin(losses,axis=0)


def simulate_person(job):
    half, source_visit, target_visit, subject, index, count, graph, w0, b0, *settings = job
    budgets = settings[0] if settings else BUDGETS
    mapping = pilot.input_map(len(graph),key=10)
    net = pilot.reservoir(graph,previous.GAIN)
    ident = 10000 + (target_visit-1)*count + index
    test = pilot.combine([pilot.simulate(net,mapping,ident,61,512,f) for f in range(FOLDS)])
    variance = test[1].var(axis=0)
    initial = (residuals(*test,w0,b0)**2).mean(axis=0)
    rows = []
    delays = []
    for rep in range(REPEATS):
        stream_length = max(BUDGETS[-1], max(budgets))//FOLDS
        streams = [pilot.simulate(net,mapping,ident,60,stream_length,FOLDS*rep+f)
                   for f in range(FOLDS)]
        for budget in budgets:
            if budget:
                four = [(x[:budget//FOLDS],y[:budget//FOLDS]) for x,y in streams]
                selections = choose(four,w0,b0)
                final = ResidualFit(*pilot.combine(four),w0,b0)
                squared_error = final.test_errors(*test,w0,b0,selections)
            else:
                selections = np.zeros(len(METHODS),dtype=int)
                squared_error = initial
            nmse = squared_error / pilot.SIGNAL_VAR
            r2 = 1-squared_error/variance
            assert np.isfinite(nmse).all() and np.isfinite(r2).all()
            for m,method in enumerate(METHODS):
                k = selections[m]
                alpha = "keep" if k==0 else format(ALPHAS[k-1],".0e")
                rows.append((half,source_visit,target_visit,subject,rep,budget,method,
                             r2[m].mean(),nmse[m].mean(),alpha))
                for d,lag in enumerate(pilot.DELAYS):
                    delays.append((half,source_visit,target_visit,subject,budget,method,lag,r2[m,d],nmse[m,d]))
    return rows, delays


def priors(graphs, half, sv, tv, subjects, previous_alphas, previous_pool, donor_valid):
    count = len(subjects)
    mapping = pilot.input_map(graphs.shape[-1],key=10)
    scans = [previous.fit_scan(g,mapping,p,sv,count) for p,g in enumerate(graphs[sv])]
    personal = previous_alphas[(previous_alphas.half == half)&(previous_alphas.source_visit == sv+1)]
    pool = previous_pool[(previous_pool.half == half)&(previous_pool.source_visit == sv+1)]
    dv = donor_valid[(donor_valid.half==half)&(donor_valid.source_visit==sv+1)]
    jobs = []
    discrepancy = 0.
    for j,name in enumerate(subjects):
        own_ref = personal[personal.target_subject == name].sort_values("delay")
        pool_ref = pool[pool.target_subject == name].sort_values("delay")
        # Reconstruct alpha from validation summaries after removing this identity.
        curve = dv[dv.subject!=name].groupby("relative_ridge").validation_nmse.mean()
        own_alpha = curve.idxmin()
        assert len(own_ref)==D and np.allclose(own_ref.transfer_alpha,own_alpha,rtol=1e-12,atol=0)
        own_model = scans[j].model
        pool_model, pool_alpha,_ = previous.shared_pool(scans,j,subjects)
        assert len(pool_ref)==D and np.allclose(pool_ref.pooled_alpha,pool_alpha,rtol=1e-12,atol=0)
        ow = own_model.weights(own_alpha)
        pw = pool_model.weights(pool_alpha)
        w0 = np.stack([ow,pw,np.zeros_like(pw)],axis=1)
        b0 = np.stack([own_model.my-own_model.mx@ow,pool_model.my-pool_model.mx@pw,
                       np.zeros_like(own_model.my)])
        target = graphs[tv,j]
        ident = 10000+tv*count+j
        net = pilot.reservoir(target,previous.GAIN)
        old_test = pilot.combine([pilot.simulate(net,mapping,ident,2,512,f) for f in range(FOLDS)])
        old_nmse = (residuals(*old_test,w0,b0)**2).mean(axis=0)/pilot.SIGNAL_VAR
        refs = np.stack([own_ref.normalized_mse,pool_ref.normalized_mse])
        difference = np.max(np.abs(old_nmse[:2]-refs))
        assert difference < 1e-10
        discrepancy = max(discrepancy,difference)
        jobs.append((half,sv+1,tv+1,name,j,count,target,w0,b0))
    return jobs,discrepancy


def check():
    """Compare the moment-based CV values with direct predictions on a fixture."""
    rng=np.random.default_rng(14)
    for n,p,rank in [(6,12,4),(12,12,7),(100,12,12)]:
        x = rng.normal(size=(n,rank)) @ rng.normal(size=(rank,p))
        xv = rng.normal(size=(16,rank)) @ rng.normal(size=(rank,p))
        y,yv = rng.normal(size=(n,D)),rng.normal(size=(16,D))
        w0=rng.normal(size=(p,len(METHODS),D)); b0=rng.normal(size=(len(METHODS),D))
        fitted=ResidualFit(x,y,w0,b0)
        grid=fitted.validation_mse(xv,yv,w0,b0)
        for m in range(len(METHODS)):
            base = (xv @ w0[:,m])+b0[m]
            r = y-(x@w0[:,m]+b0[m])
            xc=x-x.mean(axis=0)
            u,s,v=np.linalg.svd(xc,full_matrices=False)
            reference = np.mean((yv-base)**2)
            np.testing.assert_allclose(grid[0,m],reference,rtol=1e-11)
            for i,a in enumerate(ALPHAS):
                coeff = v.T @ ((s/(s*s+(s@s)/p*a))[:,None]*(u.T@(r-r.mean(axis=0))))
                prediction=base+r.mean(axis=0)+(xv-x.mean(axis=0)) @ coeff
                direct = np.mean((yv-prediction)**2,axis=0)
                np.testing.assert_allclose(grid[i+1,m],direct.mean(),rtol=1e-8,atol=1e-8)
                result=fitted.test_errors(xv,yv,w0,b0,[i+1]*len(METHODS))
                np.testing.assert_allclose(result[m],direct,rtol=1e-8,atol=1e-8)


def run(workers, budgets=BUDGETS, condition=None, out=OUT):
    check()
    budgets = tuple(sorted(set(budgets)))
    if not budgets or any(b<0 or b%FOLDS for b in budgets):
        raise ValueError("Use zero or positive multiples of four.")
    if out==OUT and (budgets!=BUDGETS or condition is not None):
        raise ValueError("Provide a separate --out for a subset or new calibration grid.")
    raw,halves,sessions,subjects,_=previous.load(previous.DEFAULT_DATA)
    path=previous.OUT
    old_personal=pd.read_csv(path/"transfer-tuned"/"pair_scores.csv",
                             dtype={"target_subject":str,"source_subject":str})
    old_personal=old_personal[old_personal.target_subject==old_personal.source_subject]
    old_pool=pd.read_csv(path/"pooled_scores.csv",dtype={"target_subject":str})
    donor_valid=pd.read_csv(path/"transfer-tuned"/"donor_validation.csv",dtype={"subject":str})
    keys=["half","source_visit","target_visit","subject"]
    rows=[]; lag_frames=[]; discrepancy=0.
    out.mkdir(exist_ok=True,parents=True)
    conditions = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for h,half in enumerate(halves):
            for sv,tv in [(0,1),(1,0)]:
                name = half + ("-forward" if sv==0 else "-reverse")
                if condition is not None and condition!=name:
                    continue
                conditions+=1
                jobs,qa = priors(raw[h],half,sv,tv,subjects,old_personal,old_pool,donor_valid)
                jobs = [(*job,budgets) for job in jobs]
                discrepancy=max(discrepancy,qa)
                start=time.monotonic()
                for done, (score,lag) in enumerate(pool.map(simulate_person,jobs),1):
                    rows.extend(score)
                    lag_cols=keys+["budget","method","delay"]
                    df=pd.DataFrame(lag,columns=lag_cols+["r2","nmse"])
                    lag_frames.append(df.groupby(lag_cols,as_index=False)[["r2","nmse"]].mean())
                    if done in (1,10,25):
                        print(f"{half} {sv+1}->{tv+1}: {done}/{len(jobs)} subjects ({time.monotonic()-start:.1f}s)",flush=True)
    df=pd.DataFrame(rows,columns=keys+["repeat","budget","method","r2","nmse","alpha"])
    lags=pd.concat(lag_frames,ignore_index=True)
    assert len(df)==conditions*25*REPEATS*len(budgets)*len(METHODS) and len(lags)==conditions*25*len(budgets)*len(METHODS)*D
    assert not df.duplicated(keys+["repeat","budget","method"]).any()
    assert not lags.duplicated(keys+["budget","method","delay"]).any()
    df.to_csv(out/"scores.csv",index=False)
    lags.to_csv(out/"delay_scores.csv",index=False)
    metadata=dict(subjects=subjects.tolist(),visits=sessions.tolist(),halves=halves.tolist(),
                  budgets=budgets,condition=condition or "all",repeats=REPEATS,folds=FOLDS,methods=METHODS,
                  max_historical_zero_budget_nmse_discrepancy=discrepancy,
                  calibration_role=60,test_role=61,prior_reference_test_role=2,
                  test_streams=4,test_stream_length=512,warmup=pilot.WARMUP,
                  alpha_grid_order=["keep",*ALPHAS.tolist()],selection="Four-stream CV inside each labeled budget.",
                  target_intercept="At positive budgets and finite ridge, learn an unpenalized intercept from the target data.",
                  simulation="Prior study acquisition RNG keys; calibration prefixes on independent streams (4*r+f), fixed test streams 0..3.",
                  counts="One labeled time step has ten previous inputs as response targets. Four independent 200-step warmups are additional unscored inputs.")
    (out/"run_info.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--condition",choices=["odd-forward","odd-reverse","even-forward","even-reverse"])
    parser.add_argument("--budgets",nargs="+",type=int,default=BUDGETS)
    args=parser.parse_args()
    if args.check:
        check()
        print("Direct-prediction and validation-moment fixtures passed.")
    else:
        run(args.workers,args.budgets,args.condition,args.out)
