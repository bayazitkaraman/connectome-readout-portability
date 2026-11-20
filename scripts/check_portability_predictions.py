"""Compare held-out predictions of the archived equal-donor scan-swap models."""
from itertools import groupby

import paired_pool_augmentation as augmentation
from nki_fewshot import ResidualFit, ALPHAS, FOLDS, REPEATS, D
import numpy as np
import pandas as pd


ROOT = augmentation.ROOT
OUT = ROOT / "results/nki-split-half"
pilot = augmentation.pilot


def score_lookup(cohort, budgets):
    folder = ROOT / "results" / (cohort + "-augmentation")
    keys = ["subject","other_subject","candidate_draw","repeat","budget","method"]
    frames = [pd.read_csv(path, dtype={"subject":str,"other_subject":str})
              for path in (folder/"scores.csv", folder/"refinement/scores.csv")]
    data = pd.concat(frames).drop_duplicates(keys+["mix_rule"])
    data = data[(data.mix_rule == "equal_donor") & data.budget.isin(budgets)]
    return data.set_index(keys)[["alpha","r2"]].to_dict("index")


def audit(cohort):
    graphs, ids, half = augmentation.read_graphs(cohort)
    jobs, _ = augmentation.make_jobs(graphs, ids, half)
    retained = [elem for elem in jobs if elem[2] == "equal_donor"]
    budgets = [0, 8, 32, 48, 64] if cohort == "nki" else [0, 8, 32, 64]
    archived = score_lookup(cohort, budgets)
    mapping = pilot.input_map(graphs.shape[-1],key=10)
    observed = []
    max_score_error = 0.
    for subject, target_group in groupby(retained, key=lambda elem: elem[3][3]):
        group = list(target_group)
        j = group[0][3][4]
        net = pilot.reservoir(graphs[1,j], .9)
        ident = 10000 + len(ids) + j
        xt, yt = pilot.combine([pilot.simulate(net,mapping,ident,61,512,f) for f in range(FOLDS)])
        variance = yt.var(axis=0)
        for rep in range(REPEATS):
            streams = [pilot.simulate(net,mapping,ident,60,588,rep*FOLDS+f) for f in range(FOLDS)]
            for budget in budgets:
                if budget:
                    x,y = pilot.combine([(a[:budget//FOLDS], b[:budget//FOLDS]) for a,b in streams])
                for draw, donor, _, job in group:
                    w0,b0 = job[-2:]
                    pred = (xt @ w0.reshape(len(w0),-1)).reshape(len(xt),3,D) + b0
                    if budget:
                        residual = ResidualFit(x,y,w0,b0)
                        coords = residual.coordinates(xt)
                    for i,method in enumerate(augmentation.METHODS):
                        row=archived[(subject,donor,draw,rep,budget,method)]
                        if row["alpha"] != "keep":
                            k, = np.flatnonzero(np.isclose(ALPHAS,float(row["alpha"]),atol=0,rtol=1e-12))
                            pred[:,i] += residual.mr[i] + coords @ residual.coef[:,k,i*D:(i+1)*D]
                        r2 = np.mean(1 - np.mean((pred[:,i]-yt)**2,axis=0)/variance)
                        max_score_error = max(max_score_error, abs(r2-row["r2"]))
                    # Root mean squared prediction discrepancy in units of target standard deviations.
                    for first, second in [(0,1),(0,2),(1,2)]:
                        distance = np.sqrt(np.mean((pred[:,first]-pred[:,second])**2 / variance))
                        observed.append(dict(cohort=cohort,subject=subject,donor=donor,repeat=rep,budget=budget,
                                             first=augmentation.METHODS[first],second=augmentation.METHODS[second],
                                             rms_prediction_difference=distance))
    assert max_score_error < 1e-8, max_score_error
    frame = pd.DataFrame(observed)
    summary = frame.groupby(["cohort","budget","first","second"]).rms_prediction_difference.agg(
        comparisons="size", mean="mean", median="median", maximum="max", exactly_equal=lambda s:(s==0).sum()
    ).reset_index()
    summary["max_archived_score_discrepancy"] = max_score_error
    return summary


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([audit(cohort) for cohort in ("nki","bnu1")]).to_csv(OUT/"prediction_convergence.csv",index=False)
