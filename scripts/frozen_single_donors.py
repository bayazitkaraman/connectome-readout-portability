"""At zero target labels, compare single first-scan decoders over all other identities."""
import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"

import json
from functools import lru_cache

import nki_transfer as transfer
from paired_pool_augmentation import read_graphs
import numpy as np
import pandas as pd


pilot = transfer.pilot
OUT = transfer.ROOT / "results/density-controls/frozen"


def target_data(graphs):
    mapping = pilot.input_map(graphs.shape[-1],key=10)
    n = graphs.shape[1]
    validation, tests = [], []
    for i, graph in enumerate(graphs[1]):
        target = pilot.reservoir(graph,transfer.GAIN)
        ident = 10000+n+i
        validation.append(pilot.simulate(target,mapping,ident,1,pilot.VALID_N))
        tests.append(pilot.combine([pilot.simulate(target,mapping,ident,61,pilot.TEST_N,f)
                                    for f in range(pilot.TEST_STREAMS)]))
    return mapping, validation, tests


def run_cohort(cohort):
    graphs, ids, half = read_graphs(cohort)
    n = len(ids)
    mapping, validation, tests = target_data(graphs)
    results, checks = [], []
    old = pd.read_csv(transfer.ROOT/f"results/{cohort}-calibration/scores.csv",dtype={"subject":str})
    previous = old[(old.source_visit==1)&(old.target_visit==2)&(old.half==half)&
                   (old.method=="personal")&(old.budget==0)].groupby("subject").r2.mean()
    for rule in ("original_distinct_inputs","identical_source_input"):
        models = []
        for i,graph in enumerate(graphs[0]):
            net = pilot.reservoir(graph,transfer.GAIN)
            ident, role = (10000+i,0) if rule=="original_distinct_inputs" else (29000,80)
            source = pilot.simulate(net,mapping,ident,role,pilot.FIT_N)
            models.append(pilot.RidgeFamily.from_design(source))
        losses = np.array([[transfer.errors(source.predict(val[0],a),val)[0].mean()
                            for a in transfer.PENALTIES] for source,val in zip(models,validation)])

        @lru_cache(None)
        def score(target,source,alpha):
            model = models[source]
            test = tests[target]
            prediction = model.predict(test[0],alpha)
            return transfer.errors(prediction,test)[1].mean()

        for t, subject in enumerate(ids):
            keep = np.arange(n) != t
            selected = transfer.PENALTIES[np.argmin(losses[keep].mean(axis=0))]
            if rule=="original_distinct_inputs":
                checks.append(abs(previous.loc[subject] - score(t,t,selected)))
            for d, donor in enumerate(ids):
                if t == d:
                    continue
                pair_mask = keep & (np.arange(n) != d)
                assert pair_mask.sum() == n-2
                alpha = transfer.PENALTIES[np.argmin(losses[pair_mask].mean(axis=0))]
                own, other = score(t,t,alpha), score(t,d,alpha)
                results.append(dict(cohort=cohort,source_input=rule,target_id=subject,other_id=donor,
                                    relative_penalty=alpha,own_r2=own,other_r2=other,gain=own-other))
        score.cache_clear()
    assert max(checks)<1e-8
    return results, max(checks)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    parts,check = [],{}
    for cohort in ("nki","bnu1"):
        records, error = run_cohort(cohort)
        parts.extend(records)
        check[cohort] = float(error)
    pairs = pd.DataFrame(parts)
    people = pairs.groupby(["cohort","source_input","target_id"])[["own_r2","other_r2","gain"]].mean().reset_index()
    summary=[]
    for (cohort,rule),g in people.groupby(["cohort","source_input"]):
        n=len(g)
        blk = pairs[(pairs.cohort==cohort)&(pairs.source_input==rule)]
        assert len(blk)==n*(n-1)
        weights = np.random.default_rng(20261120).multinomial(n,np.ones(n)/n,size=10000)
        gains = weights @ g.gain.to_numpy()/n
        lo,hi=np.quantile(gains,[.025,.975])
        summary.append(dict(cohort=cohort,source_input=rule,participants=n,donors_per_target=n-1,
                            ordered_comparisons=len(blk),mean_own_r2=g.own_r2.mean(),
                            mean_other_r2=g.other_r2.mean(),mean_gain=g.gain.mean(),gain_ci_low=lo,gain_ci_high=hi,
                            people_favoring_own=int((g.gain>0).sum())))
    pairs.to_csv(OUT/"single_donor_pairs.csv",index=False)
    people.to_csv(OUT/"single_donor_participant_means.csv",index=False)
    pd.DataFrame(summary).to_csv(OUT/"summary.csv",index=False)
    (OUT/"verification.json").write_text(json.dumps(dict(max_difference_from_prior_leave_one_own_scores=check,
        primary_source_role=0,matched_input_sensitivity_role=80,target_role=61,tuning_validation_role=1,
        source_rows=2352,target_labels=0,target_test_steps=2048,bootstrap="10,000 target participants; full donor bank fixed"),
        indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(summary).round(6).to_string(index=False))


if __name__=="__main__":
    main()
