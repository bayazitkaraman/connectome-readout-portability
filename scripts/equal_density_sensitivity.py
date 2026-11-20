"""Refit the matched donor swap after retaining an equal edge count within each cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import paired_pool_augmentation as aug
import analyze_anatomy_advantage as anatomy
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr


OUT = aug.ROOT / "results/density-controls/equal-density"
SEEDS = (61973, 61974)
EDGE_COUNTS = {"nki": 177, "bnu1": 1291}
LOW = (0, 8)


def thin_graphs(graphs, subjects, cohort, seed):
    n = graphs.shape[-1]
    upper = np.triu_indices(n, k=1)
    values = graphs[..., upper[0], upper[1]]
    k = int((values > 0).sum(axis=-1).min())
    assert k == EDGE_COUNTS[cohort]
    priority = np.random.default_rng(np.random.SeedSequence([seed, len(subjects)])).permutation(len(upper[0]))
    mapping = aug.pilot.input_map(n, key=10)
    read = mapping[1]
    inputs = np.flatnonzero(mapping[0].ravel())
    retain = np.zeros_like(graphs)
    qc = []
    for visit in range(2):
        for i, subject in enumerate(subjects):
            v = values[visit,i].astype(float)
            chosen = np.lexsort((priority, -v))[:k]
            cutoff = v[chosen[-1]]
            a, b = upper[0][chosen], upper[1][chosen]
            g = retain[visit,i]
            g[a,b] = g[b,a] = v[chosen]
            assert cutoff > 0 and np.count_nonzero(np.triu(g,1)) == k
            groups, component = connected_components(g, directed=False)
            reachable = np.isin(component[read], component[inputs])
            states, _ = aug.pilot.simulate(aug.pilot.reservoir(g,aug.transfer.GAIN),
                                           mapping,29000,80,aug.pilot.FIT_N)
            qc.append(dict(subject=subject, visit=visit+1, retained_edges=k, original_edges=int((v>0).sum()),
                           retained_weight_fraction=float(v[chosen].sum()/v.sum()),
                           cutoff=cutoff, cutoff_tie_split=int((v==cutoff).sum())>int((v[chosen]==cutoff).sum()),
                           original_isolates=int(((graphs[visit,i]>0).sum(axis=0)==0).sum()),
                           retained_isolates=int((g.sum(axis=0)==0).sum()), connected_components=groups,
                           reachable_readouts=int(reachable.sum()), active_readouts=int((states.std(axis=0)>0).sum()),
                           state_rms=float(np.sqrt(np.mean(states**2)))))
    np.testing.assert_array_equal(retain, retain.swapaxes(-1,-2))
    return retain, pd.DataFrame(qc)


def target_scores(jobs):
    # Reuse target streams across the six pair/allocation fits for each person.
    first = jobs[0][3]
    _, _, _, person, idx, n, graph, w, b = first
    pipeline = aug.calibration
    mapping = aug.pilot.input_map(graph.shape[-1], key=10)
    net = aug.pilot.reservoir(graph,aug.transfer.GAIN)
    key = 10000 + n + idx
    tx, ty = aug.pilot.combine([aug.pilot.simulate(net,mapping,key,61,512,f) for f in range(4)])
    test_var = ty.var(axis=0)
    blank_w, blank_b = np.zeros_like(w), np.zeros_like(b)
    rows, ceiling = [], []
    for rep in range(6):
        streams = [aug.pilot.simulate(net,mapping,key,60,588,4*rep+f) for f in range(4)]
        fitted = pipeline.ResidualFit(*aug.pilot.combine(streams), blank_w, blank_b)
        choices = pipeline.choose(streams, blank_w, blank_b)
        errors = fitted.test_errors(tx,ty,blank_w,blank_b,choices)
        ceiling.append(dict(subject=person,repeat=rep, target_only_r2_2352=float((1-errors[0]/test_var).mean())))
        early = [(x[:2], y[:2]) for x,y in streams]
        for draw, other, rule, job in jobs:
            assert job[3] == person
            slopes, intercept = job[-2:]
            initial = (pipeline.residuals(tx,ty,slopes,intercept)**2).mean(axis=0)
            choices = pipeline.choose(early,slopes,intercept)
            fitted = pipeline.ResidualFit(*aug.pilot.combine(early),slopes,intercept)
            errors = fitted.test_errors(tx,ty,slopes,intercept,choices)
            for budget, err in zip(LOW, (initial, errors)):
                r2 = (1-err/test_var).mean(axis=1)
                for method,score in zip(aug.METHODS,r2):
                    rows.append(dict(subject=person,candidate_draw=draw,other_subject=other,mix_rule=rule,
                                     repeat=rep,budget=budget,method=method,r2=score))
    return rows, ceiling


def summaries(scores, subjects, graphs, full_scores):
    n = len(subjects)
    ids = {s:i for i,s in enumerate(subjects)}
    u,v = np.triu_indices(graphs.shape[-1],1)
    vectors = graphs[...,u,v]
    similarities = np.corrcoef(vectors.reshape(2*n, -1))[n:,:n]
    keys = ["mix_rule","subject","candidate_draw","other_subject","budget"]
    pairs = scores.groupby(keys + ["method"]).r2.mean().unstack().reset_index()
    own = np.array([ids[s] for s in pairs.subject])
    other = np.array([ids[s] for s in pairs.other_subject])
    pairs["anatomy_advantage"] = similarities[own,own] - similarities[own,other]
    pairs["accuracy_advantage"] = pairs.pool_plus_own - pairs.pool_plus_other
    out = []
    for (rule,budget), frame in pairs.groupby(["mix_rule","budget"]):
        per = frame.groupby("subject")[["anatomy_advantage","accuracy_advantage","pool_plus_own","pool_plus_other"]].mean()
        x = per.anatomy_advantage.to_numpy()
        y = per.accuracy_advantage.to_numpy()
        draws = np.random.default_rng(anatomy.BOOT_SEED).integers(0,n,size=(10_000,n))
        rho = spearmanr(x,y).statistic
        ci = np.nanquantile(anatomy.rank_correlations(x[draws],y[draws]),[.025,.975])
        yi = np.quantile(y[draws].mean(axis=1),[.025,.975])
        out.append(dict(mix_rule=rule, budget=budget, own_r2=per.pool_plus_own.mean(),
                        other_r2=per.pool_plus_other.mean(), mean_gain=y.mean(),gain_ci_low=yi[0],
                        gain_ci_high=yi[1], rho=rho,rho_ci_low=ci[0],rho_ci_high=ci[1],
                        target_only_r2_2352=full_scores.target_only_r2_2352.mean()))
    return pairs, pd.DataFrame(out)


def run(cohort, seed, workers):
    graphs, people, half = aug.read_graphs(cohort)
    source, qc = thin_graphs(graphs,people,cohort,seed)
    jobs, _ = aug.make_jobs(source, people, half)
    batches = [[job for job in jobs if job[3][3]==person] for person in people]
    assert all(len(batch)==6 for batch in batches)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(target_scores,batches))
    scores = pd.DataFrame([row for batch in outcomes for row in batch[0]])
    full = pd.DataFrame([row for batch in outcomes for row in batch[1]])
    assert len(scores)==len(people)*6*3*2*6 and len(full)==len(people)*6
    direct,_ = aug.score_one((*jobs[0],LOW))
    ref = scores[(scores.subject==people[0]) & (scores.candidate_draw==0) &
                 (scores.mix_rule==jobs[0][2])]
    keys = ["repeat","budget","method"]
    audit = direct.merge(ref,on=keys,suffixes=("_pipeline","_cached"),validate="one_to_one")
    np.testing.assert_allclose(audit.r2_pipeline,audit.r2_cached,rtol=1e-10,atol=1e-10)
    pairs, result = summaries(scores,people,source,full)
    for data in (qc,full,scores,pairs,result):
        data.insert(0,"tie_seed",seed)
        data.insert(0,"cohort",cohort)
    folder = OUT / f"{cohort}_{seed}"
    folder.mkdir(exist_ok=True,parents=True)
    for name, data in [("scans",qc),("target_only",full),("scores",scores),("pairs",pairs),("summary",result)]:
        data.to_csv(folder / f"{name}.csv",index=False)
    print(result.round(5).to_string(index=False),flush=True)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--cohort",nargs="+",choices=tuple(EDGE_COUNTS),default=list(EDGE_COUNTS))
    parser.add_argument("--seed",nargs="+",type=int,default=list(SEEDS))
    args=parser.parse_args()
    for seed in args.seed:
        for cohort in args.cohort:
            print("Scoring",cohort,seed,flush=True)
            run(cohort,seed,args.workers)
