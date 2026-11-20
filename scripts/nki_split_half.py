"""Compare anatomical identification and zero-label single-decoder transfer across NKI diffusion halves."""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import nki_transfer as transfer
import numpy as np
import pandas as pd
from scipy.stats import rankdata


pilot = transfer.pilot
OUT = transfer.ROOT / "results/nki-split-half"
KEY = ["src_visit", "src_half", "target_visit", "target_half"]
REPS = 10_000
SEED = 20261120


@dataclass
class Scan:
    model: object
    validation: tuple
    test: tuple
    reference_r2: float


def build_cell(graphs, visit, mapping):
    scans = []
    for index, graph in enumerate(graphs):
        ident = 10000 + (visit-1)*len(graphs) + index
        net = pilot.reservoir(graph, transfer.GAIN)
        train = pilot.simulate(net, mapping, ident, 0, pilot.FIT_N)
        validation = pilot.simulate(net, mapping, ident, 1, pilot.VALID_N)
        test = pilot.combine([pilot.simulate(net,mapping,ident,61,pilot.TEST_N,k)
                              for k in range(pilot.TEST_STREAMS)])
        model = pilot.RidgeFamily.from_design(train)
        alpha = model.tune(validation, transfer.PENALTIES)
        ref_r2 = transfer.errors(model.predict(test[0],alpha), test)[1].mean()
        scans.append(Scan(model,validation,test,ref_r2))
    return scans


def select_pair_penalties(sources, targets):
    losses = np.array([[transfer.errors(source.model.predict(target.validation[0],alpha),
                                         target.validation)[0].mean()
                        for alpha in transfer.PENALTIES] for source,target in zip(sources,targets)])
    n = len(sources)
    selected = np.full((n,n), np.nan)
    for t in range(n):
        for d in range(t):
            keep = (np.arange(n) != t) & (np.arange(n) != d)
            alpha = transfer.PENALTIES[losses[keep].mean(axis=0).argmin()]
            selected[t,d] = selected[d,t] = alpha
    return selected


def score_condition(source_key,target_key,cells,edges,selections,subjects):
    sv,sh = source_key
    tv,th = target_key
    source, target = cells[source_key], cells[target_key]
    similarities = edges[target_key] @ edges[source_key].T
    selected = selections[source_key, target_key]
    # Within-visit sensitivity: use the prior penalties selected when the source
    # is from the other visit on the same half and target is exactly unchanged.
    alt = selections[(3-sv,sh),target_key] if sv == tv else selected

    @lru_cache(None)
    def heldout(t,d,alpha):
        test = target[t].test
        pred = source[d].model.predict(test[0],alpha)
        return transfer.errors(pred,test)[1].mean()

    result = []
    for t,subject in enumerate(subjects):
        for d,other in enumerate(subjects):
            if t == d:
                continue
            alpha = selected[t,d]
            own, foreign = heldout(t,t,alpha), heldout(t,d,alpha)
            alt_own, alt_foreign = heldout(t,t,alt[t,d]), heldout(t,d,alt[t,d])
            result.append(dict(src_visit=sv,src_half=sh,target_visit=tv,target_half=th,
                               target_id=subject,other_id=other,alpha=alpha,
                               own_r2=own,other_r2=foreign,gain=own-foreign,
                               own_similarity=similarities[t,t],other_similarity=similarities[t,d],
                               anatomy_advantage=similarities[t,t]-similarities[t,d],
                               reference_r2=target[t].reference_r2,alt_alpha=alt[t,d],
                               alt_own_r2=alt_own,alt_other_r2=alt_foreign,alt_gain=alt_own-alt_foreign))
    return result


def summarize_participants(pairs):
    metrics = ["own_r2","other_r2","gain","own_similarity","other_similarity","anatomy_advantage",
               "reference_r2","alt_own_r2","alt_other_r2","alt_gain"]
    frame = pairs.copy()
    frame["anatomically_better_stranger"] = frame.anatomy_advantage < -1e-12
    frame["anatomically_tied_stranger"] = frame.anatomy_advantage.abs() <= 1e-12
    frame["decoding_win"] = frame.gain > 0
    people = frame.groupby(KEY+["target_id"]).agg(
        {**{name:"mean" for name in metrics},"anatomically_better_stranger":"sum",
         "anatomically_tied_stranger":"sum","decoding_win":"sum"}).reset_index()
    people["anatomic_rank"] = 1 + people.anatomically_better_stranger + .5*people.anatomically_tied_stranger
    people["anatomic_top1"] = (people.anatomically_better_stranger + people.anatomically_tied_stranger) == 0
    return people


def percentile_rank_corr(x,y):
    a,b = rankdata(x,axis=-1), rankdata(y,axis=-1)
    a-=a.mean(axis=-1,keepdims=True)
    b-=b.mean(axis=-1,keepdims=True)
    norm = np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1))
    return np.divide(np.sum(a*b,axis=-1),norm,out=np.full(norm.shape,np.nan),where=norm>0)


def mean_interval(x,draws):
    x = np.asarray(x)
    return [x.mean(), *np.percentile(x[draws].mean(axis=1),[2.5,97.5])]


def append_mean(row,label,x,draws):
    row[label],row[label+"_lo"],row[label+"_hi"] = mean_interval(x,draws)


def summarize_conditions(people,draws):
    rows = []
    for condition,p in people.groupby(KEY):
        p = p.sort_values("target_id")
        assert len(p) == draws.shape[1]
        row = dict(zip(KEY,condition))
        for metric in ["own_r2","other_r2","gain","own_similarity","other_similarity","anatomy_advantage",
                       "reference_r2","alt_own_r2","alt_gain"]:
            append_mean(row,metric,p[metric],draws)
        x,y = p.anatomy_advantage.to_numpy(),p.gain.to_numpy()
        row["rho_anatomy_gain"] = float(percentile_rank_corr(x,y))
        row["rho_lo"],row["rho_hi"] = np.nanpercentile(percentile_rank_corr(x[draws],y[draws]),[2.5,97.5])
        row.update(n=len(p),own_beats_mean_other=int((p.gain>0).sum()),
                   anatomy_beats_mean_other=int((p.anatomy_advantage>1e-12).sum()),
                   anatomy_unique_top1=int(p.anatomic_top1.sum()),median_anatomy_rank=p.anatomic_rank.median(),
                   median_own_r2=p.own_r2.median(),median_gain=p.gain.median(),
                   fraction_pair_wins=p.decoding_win.mean()/(len(p)-1))
        rows.append(row)
    return pd.DataFrame(rows)


def matched_visit_contrasts(people,draws):
    rows=[]
    for (tv,th), target in people.groupby(["target_visit","target_half"]):
        sh = "odd" if th=="even" else "even"
        same=target[(target.src_visit==tv)&(target.src_half==sh)].set_index("target_id").sort_index()
        other=target[(target.src_visit!=tv)&(target.src_half==sh)].set_index("target_id").sort_index()
        assert same.index.equals(other.index)
        row=dict(target_visit=tv,target_half=th,source_half=sh,n=len(same))
        for metric in ["own_r2","other_r2","gain","own_similarity","anatomy_advantage","alt_own_r2","alt_gain"]:
            append_mean(row, "same_minus_cross_"+metric, same[metric]-other[metric], draws)
        row["people_higher_own_r2_same_visit"]=int((same.own_r2>other.own_r2).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    raw,halves,_,subjects,_=transfer.load(transfer.DEFAULT_DATA)
    mapping=pilot.input_map(raw.shape[-1],key=10)
    ii,jj=np.triu_indices(raw.shape[-1],1)
    cells,edges={},{}
    for h,half in enumerate(halves):
        for v in range(2):
            key=(v+1,half)
            graphs=raw[h,v]
            x=graphs[:,ii,jj]
            x-=x.mean(axis=1,keepdims=True)
            edges[key]=x/np.linalg.norm(x,axis=1,keepdims=True)
            cells[key]=build_cell(graphs,v+1,mapping)
    selections={(a,b):select_pair_penalties(cells[a],cells[b]) for a,b in permutations(cells,2)}
    records=[]
    for a,b in permutations(cells,2):
        records.extend(score_condition(a,b,cells,edges,selections,subjects))
    pairs=pd.DataFrame(records)
    assert len(pairs)==12*25*24 and not pairs.duplicated(KEY+["target_id","other_id"]).any()
    archived=pd.read_csv(transfer.ROOT/"results/density-controls/frozen/single_donor_pairs.csv")
    archived=archived[(archived.cohort=="nki")&(archived.source_input=="original_distinct_inputs")]
    replay=pairs[(pairs.src_visit==1)&(pairs.src_half=="odd")&(pairs.target_visit==2)&(pairs.target_half=="odd")]
    joined=replay.merge(archived,on=["target_id","other_id"],validate="one_to_one")
    assert len(joined)==600 and (joined.alpha==joined.relative_penalty).all()
    error = max((joined[name+"_x"]-joined[name+"_y"]).abs().max() for name in ("own_r2","other_r2","gain"))
    assert error<1e-9
    people=summarize_participants(pairs)
    draws=np.random.default_rng(SEED).integers(len(subjects),size=(REPS,len(subjects)))
    summary=summarize_conditions(people,draws)
    contrasts=matched_visit_contrasts(people,draws)
    OUT.mkdir(exist_ok=True,parents=True)
    for name,table in (("pair_scores",pairs),("subject_scores",people),("summary",summary),("matched_visit_contrasts",contrasts)):
        table.to_csv(OUT/(name+".csv"),index=False)
    metadata=dict(input_sha256=hashlib.sha256(transfer.DEFAULT_DATA.read_bytes()).hexdigest(),
                  n_subjects=len(subjects),ordered_cells=len(summary),pairs_per_condition=600,
                  source_training_steps=pilot.FIT_N,donor_penalty_validation_steps_per_identity=pilot.VALID_N,
                  target_transfer_labels=0,test_steps=pilot.TEST_STREAMS*pilot.TEST_N,
                  reference_training_steps=pilot.FIT_N,reference_validation_steps=pilot.VALID_N,
                  source_role=0,validation_role=1,test_role=61,stream_id="10000+(visit-1)*25+subject_index",
                  regularization="Transfer selected on remaining 23 own source-to-target validation pairs; same penalty applied to both tested decoders",
                  alt_regularization="Same target and source half, cross-visit donors as tuning comparison for within-visit scores",
                  bootstrap="Paired resamples of targets within condition with fixed trained fits, full foil bank, and streams",
                  bootstrap_reps=REPS,bootstrap_seed=SEED,tolerance_anatomical_ties=1e-12,archived_pair_replay_max_abs_error=error)
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(summary[KEY+["own_r2","other_r2","gain","gain_lo","gain_hi","own_beats_mean_other",
                       "own_similarity","anatomy_unique_top1","anatomy_beats_mean_other","rho_anatomy_gain",
                       "reference_r2"]].round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))


if __name__=="__main__":
    run()
