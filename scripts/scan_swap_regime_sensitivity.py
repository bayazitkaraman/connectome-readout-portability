"""Re-evaluate matched, equal-share first-scan swaps across a fixed dynamical/ridge grid."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import paired_pool_augmentation as aug
import analyze_anatomy_advantage as anatomy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUT = aug.ROOT / "results/density-controls/regime"
GAINS = (.70, .90, 1.00)
MULTIPLIERS = (.1, 1., 10.)


def prepare(graphs, people, gain):
    n = len(people)
    mapping = aug.pilot.input_map(graphs.shape[-1],key=10)
    bank, validation, candidates = [], [], []
    for i, graph in enumerate(graphs[0]):
        network = aug.pilot.reservoir(graph,gain)
        bank.append(aug.pilot.simulate(network,mapping,10000+i,0,2352))
        validation.append(aug.pilot.simulate(network,mapping,10000+i,1,1176))
        candidates.append(aug.pilot.simulate(network,mapping,29000,80,2352))

    @lru_cache(None)
    def bank_moment(i, count):
        return aug.pilot.Moment.from_data(aug.pilot.sampled(bank[i],count))

    def pooled_moment(allocation):
        return aug.pilot.merge([bank_moment(i,count) for i,count in allocation])

    added = 2352//(n-1)
    assert added*(n-1)==2352
    single = [aug.pilot.Moment.from_data(aug.pilot.sampled(data,added)) for data in candidates]
    assignments = aug.partners(n)
    tasks, packed = [], []
    for i, name in enumerate(people):
        models = []
        for draw,j in enumerate(assignments[i]):
            common = np.array([idx for idx in range(n) if idx not in (i,j)])
            shuffled = aug.pilot.rng(173,n,i,j).permutation(common)
            split = int(round(len(shuffled)*2/3))
            inner = pooled_moment(aug.split_rows(shuffled[:split],2352))
            valid = aug.pilot.combine([aug.pilot.sampled(validation[ix],count)
                                       for ix,count in aug.split_rows(shuffled[split:],1176)])
            selected = aug.pilot.RidgeFamily(inner).tune(valid,aug.transfer.PENALTIES)
            common_moments = pooled_moment(aug.split_rows(common,2352-added))
            base = [aug.pilot.RidgeFamily(aug.pilot.merge([common_moments,single[idx]])) for idx in (i,j)]
            for multiplier in MULTIPLIERS:
                slopes = [model.weights(selected*multiplier) for model in base]
                intercepts = [model.my - model.mx @ w for model,w in zip(base,slopes)]
                # Three parallel positions are required by the original residual solver.
                # The last is a dummy slot with zero prior and never used in analysis.
                slopes.append(np.zeros_like(slopes[0]))
                intercepts.append(np.zeros_like(intercepts[0]))
                models.append((draw,people[j],multiplier,np.stack(slopes,axis=1),np.stack(intercepts)))
            packed.append(dict(subject=name,other_subject=people[j],candidate_draw=draw,selected_alpha=selected))
        tasks.append((i,name,n,graphs[1,i],gain,models))
    return tasks,pd.DataFrame(packed)


def score(task):
    i, person, n, graph, gain, fits = task
    target_id = 10000+n+i
    mapping = aug.pilot.input_map(len(graph),10)
    net = aug.pilot.reservoir(graph,gain)
    xt, yt = aug.pilot.combine([aug.pilot.simulate(net,mapping,target_id,61,512,f) for f in range(4)])
    variance = yt.var(axis=0)
    rows = []
    for rep in range(6):
        seq = [aug.pilot.simulate(net,mapping,target_id,60,588,4*rep+f) for f in range(4)]
        early = [(x[:2],y[:2]) for x,y in seq]
        for draw,other,multiplier,weights,intercepts in fits:
            choices = aug.calibration.choose(early,weights,intercepts)
            fit = aug.calibration.ResidualFit(*aug.pilot.combine(early),weights,intercepts)
            mse = fit.test_errors(xt,yt,weights,intercepts,choices)
            metric = (1-mse/variance).mean(axis=1)
            rows.append(dict(subject=person,other_subject=other,candidate_draw=draw,repeat=rep,
                             multiplier=multiplier,own_r2=metric[0],other_r2=metric[1]))
    return rows


def check_reference(cohort, rows):
    reference = pd.read_csv(aug.ROOT / f"results/{cohort}-augmentation/scores.csv",
                            dtype={"subject":str, "other_subject":str})
    reference=reference[(reference.mix_rule=="equal_donor") & (reference.budget==8)]
    keys = ["subject","candidate_draw","other_subject","repeat"]
    pivot = reference.pivot(index=keys,columns="method",values="r2")
    combined = rows[rows.multiplier==1.].merge(pivot, left_on=keys,right_index=True,validate="one_to_one")
    assert len(combined)==len(pivot)
    delta = np.abs(combined[["own_r2","other_r2"]].to_numpy() -
                   combined[["pool_plus_own","pool_plus_other"]].to_numpy())
    assert delta.max()<1e-10, delta.max()
    return float(delta.max())


def run(cohort, gains, workers):
    graphs, people, _ = aug.read_graphs(cohort)
    reference = pd.read_csv(anatomy.OUT / "participants.csv",dtype={"subject":str})
    x_table = reference[(reference.cohort==cohort)&(reference.mix_rule=="equal_donor")].set_index("subject")
    destination = OUT / cohort
    destination.mkdir(exist_ok=True,parents=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for gain in gains:
            tasks, metadata = prepare(graphs,people,gain)
            rows = pd.DataFrame([item for target in pool.map(score,tasks) for item in target])
            assert len(rows)==len(people)*3*6*len(MULTIPLIERS)
            assert np.isfinite(rows[["own_r2","other_r2"]]).all().all()
            rows["accuracy_advantage"] = rows.own_r2-rows.other_r2
            rows["gain"] = gain
            rows.to_csv(destination/f"scores_{gain:.2f}.csv",index=False)
            metadata.to_csv(destination/f"penalties_{gain:.2f}.csv",index=False)
            check=check_reference(cohort,rows) if gain==.9 else np.nan
            for factor, group in rows.groupby("multiplier"):
                avg=group.groupby("subject")[["own_r2","other_r2","accuracy_advantage"]].mean()
                x = x_table.loc[avg.index,"anatomy_advantage"].to_numpy()
                y = avg.accuracy_advantage.to_numpy()
                idx=np.random.default_rng(anatomy.BOOT_SEED).integers(0,len(y),size=(10_000,len(y)))
                ci=np.quantile(y[idx].mean(axis=1),[.025,.975])
                rho=spearmanr(x,y).statistic
                ri=np.quantile(anatomy.rank_correlations(x[idx],y[idx]),[.025,.975])
                results.append(dict(cohort=cohort,gain=gain,source_ridge_multiplier=factor,
                                    own_r2=avg.own_r2.mean(),other_r2=avg.other_r2.mean(),
                                    mean_accuracy_advantage=y.mean(),ci_low=ci[0],ci_high=ci[1],
                                    rho=rho,rho_ci_low=ri[0],rho_ci_high=ri[1],
                                    baseline_score_error=check if factor==1 else np.nan))
            print(cohort,gain,pd.DataFrame(results[-3:]).round(5).to_string(index=False),flush=True)
    pd.DataFrame(results).to_csv(destination/"summary.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--cohort",nargs="+",choices=["nki","bnu1"],default=["nki","bnu1"])
    args=parser.parse_args()
    for cohort in args.cohort:
        run(cohort,GAINS,args.workers)
