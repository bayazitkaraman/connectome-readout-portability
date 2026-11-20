"""Score the same frozen personal readout before and after a wiring substitution."""
import argparse
import json
from dataclasses import dataclass
from functools import lru_cache

import nki_transfer as transfer
from paired_pool_augmentation import read_graphs
import numpy as np
import pandas as pd


pilot = transfer.pilot
ROOT = transfer.ROOT
OUT = ROOT / "results/transfer-similarity"
KEY = ["cohort", "src_visit", "src_half", "target_visit", "target_half"]


@dataclass
class Source:
    model: object
    local_alpha: float


def archived(cohort,grid):
    if cohort == "nki":
        pairs = pd.read_csv(ROOT / "results/nki-split-half/pair_scores.csv",
                            dtype={"target_id": str, "other_id": str})
        pairs["cohort"] = "nki"
        raw, halves, _, ids, _ = transfer.load(transfer.DEFAULT_DATA)
        graphs = {(v+1, half): raw[h,v] for v in range(2) for h, half in enumerate(halves)}
    else:
        pairs = pd.read_csv(ROOT / "results/density-controls/frozen/single_donor_pairs.csv",
                            dtype={"target_id": str, "other_id": str})
        pairs = pairs.loc[(pairs.cohort == cohort) & (pairs.source_input == "original_distinct_inputs")].copy()
        pairs["src_visit"], pairs["target_visit"] = 1, 2
        pairs["src_half"] = pairs["target_half"] = "all"
        pairs = pairs.rename(columns={"relative_penalty":"alpha"})
        raw, ids, _ = read_graphs(cohort)
        graphs = {(v+1,"all"): raw[v] for v in range(2)}
    if grid=="cross_tuned":
        if cohort=="nki":
            pairs["alpha"], pairs["own_r2"] = pairs.alt_alpha,pairs.alt_own_r2
    elif grid=="fixed10":
        # Literal common slope penalty; held-out scores are independently checked
        # against the archived comparison whenever that comparison also used 10.
        pairs.loc[pairs.alpha!=10, "own_r2"]=np.nan
        pairs["alpha"]=10.
    elif grid=="dense":
        dense=pd.read_csv(OUT/"grid_pairs.csv",dtype={"target_id":str,"other_id":str})
        dense=dense.loc[dense.cohort==cohort,KEY+["target_id","other_id","dense_alpha","dense_own"]]
        prior_n=len(pairs)
        pairs=pairs.merge(dense,on=KEY+["target_id","other_id"],validate="one_to_one")
        assert len(pairs)==prior_n
        pairs["alpha"], pairs["own_r2"] = pairs.dense_alpha, pairs.dense_own
    return pairs, graphs, ids


def score_cohort(cohort,grid):
    pairs, graphs, ids = archived(cohort,grid)
    n=len(ids)
    indices = {person:j for j,person in enumerate(ids)}
    mapping = pilot.input_map(next(iter(graphs.values())).shape[-1],key=10)

    @lru_cache(None)
    def source(key,j):
        net = pilot.reservoir(graphs[key][j],transfer.GAIN)
        ident = 10000+(key[0]-1)*n+j
        train = pilot.simulate(net,mapping,ident,0,pilot.FIT_N)
        validation = pilot.simulate(net,mapping,ident,1,pilot.VALID_N)
        model = pilot.RidgeFamily.from_design(train)
        return Source(model, model.tune(validation,transfer.PENALTIES))

    @lru_cache(None)
    def test(graph_key,j,target_visit):
        # Target participant AND destination visit determine stimulation on both graphs.
        ident = 10000+(target_visit-1)*n+j
        net=pilot.reservoir(graphs[graph_key][j],transfer.GAIN)
        return pilot.combine([pilot.simulate(net,mapping,ident,61,pilot.TEST_N,k)
                              for k in range(pilot.TEST_STREAMS)])

    rows = []
    for condition,block in pairs.groupby(KEY):
        _,sv,sh,tv,th = condition
        src, dest = (sv,sh),(tv,th)
        for person,person_pairs in block.groupby("target_id"):
            assert len(person_pairs)==n-1
            j=indices[person]
            src_test=test(src,j,tv)
            tgt_test=test(dest,j,tv)
            np.testing.assert_array_equal(src_test[1],tgt_test[1])
            estimator=source(src,j)
            local=estimator.local_alpha
            on_source = {}
            on_target = {}
            for alpha in set(person_pairs.alpha).union({local}):
                for data,results in [(src_test,on_source),(tgt_test,on_target)]:
                    results[alpha]=transfer.errors(estimator.model.predict(data[0],alpha),data)[1].mean()
            for pair in person_pairs.itertuples():
                alpha=pair.alpha
                if np.isfinite(pair.own_r2):
                    np.testing.assert_allclose(on_target[alpha],pair.own_r2,atol=1e-9,rtol=1e-9)
                rows.append(dict(zip(KEY,condition)) | {
                    "target_id":person, "other_id":pair.other_id, "transfer_alpha":alpha, "local_alpha":local,
                    "local_source_r2":on_source[local], "local_target_r2":on_target[local],
                    "transfer_source_r2":on_source[alpha], "transfer_target_r2":on_target[alpha],
                    "source_penalty_cost":on_source[local]-on_source[alpha],
                    "frozen_wiring_drop":on_source[alpha]-on_target[alpha],
                    "total_drop":on_source[local]-on_target[alpha]})
    return pd.DataFrame(rows)


def run(grid):
    OUT.mkdir(parents=True,exist_ok=True)
    pairs = pd.concat([score_cohort(c,grid) for c in ("nki","bnu1")],ignore_index=True)
    metrics = ["local_source_r2","local_target_r2","transfer_source_r2","transfer_target_r2",
               "source_penalty_cost","frozen_wiring_drop","total_drop"]
    assert not pairs.duplicated(KEY+["target_id","other_id"]).any()
    people = pairs.groupby(KEY+["target_id"],as_index=False)[metrics].mean()
    rows = []
    for condition, g in people.groupby(KEY):
        n=len(g)
        draws=np.random.default_rng(20261120).integers(n,size=(10_000,n))
        row=dict(zip(KEY,condition)) | {"n":n}
        for z in metrics:
            vals=g[z].to_numpy()
            row[z]=vals.mean()
            row[z+"_lo"],row[z+"_hi"]=np.percentile(vals[draws].mean(axis=1),[2.5,97.5])
            row[z+"_median"]=np.median(vals)
        row["people_frozen_drop"]=int(np.sum(g.frozen_wiring_drop>0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    np.testing.assert_allclose(summary.total_drop,summary.source_penalty_cost+summary.frozen_wiring_drop,atol=2e-12)
    prefix="source_self" if grid=="archived" else "source_self_"+grid
    for name,frame in (("pairs",pairs),("subjects",people),("summary",summary)):
        frame.to_csv(OUT/f"{prefix}_{name}.csv",index=False)
    info = dict(source_rows=pilot.FIT_N, local_validation_rows=pilot.VALID_N, target_training_labels=0,
                test_role=61, test_steps=pilot.TEST_N*pilot.TEST_STREAMS,
                personal_decoder="For each target person and held-out other identity, use the original pair-specific transfer penalty on the person's first-scan training data.",
                signals="Drive the source and target graphs with exactly the same four fresh target-visit signals. Source training and validation use separate streams.",
                confirm="Target test score checked against available pair score at the selected penalty; paired source-to-target drop calculated without changing readout weights.",
                penalty_rule=grid,
                bootstrap="10,000 paired target-subject resamples, conditional on fitted readouts and donor bank")
    (OUT/f"{prefix}_metadata.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(summary[KEY+metrics].round(5).to_string(index=False))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--grid",choices=["archived","cross_tuned","fixed10","dense"],default="archived")
    run(cli.parse_args().grid)
