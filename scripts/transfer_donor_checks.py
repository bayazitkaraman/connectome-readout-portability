"""Check selection by donor anatomy and the flagged-person sensitivity on original repeat scans."""
import json

from frozen_single_donors import pilot, target_data, transfer
from paired_pool_augmentation import read_graphs
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUT = transfer.ROOT / "results/transfer-similarity"
FLAGGED = "A00063383"


def prepare(cohort):
    graphs, ids, _ = read_graphs(cohort)
    u, v = np.triu_indices(graphs.shape[-1], 1)
    edge = graphs[:, :, u, v].astype(float)
    edge -= edge.mean(axis=-1, keepdims=True)
    edge /= np.linalg.norm(edge, axis=-1, keepdims=True)
    similarities = edge[1] @ edge[0].T
    scores = pd.read_csv(transfer.ROOT / "results/density-controls/frozen/single_donor_pairs.csv",
                         dtype={"target_id": str, "other_id": str})
    scores = scores[(scores.cohort == cohort) & (scores.source_input == "original_distinct_inputs")].copy()
    assert len(scores) == len(ids) * (len(ids)-1) and scores.relative_penalty.nunique() == 1
    lookup = {subject: i for i, subject in enumerate(ids)}
    target = scores.target_id.map(lookup).to_numpy()
    donor = scores.other_id.map(lookup).to_numpy()
    scores["own_similarity"] = similarities[target,target]
    scores["other_similarity"] = similarities[target,donor]
    return graphs, ids, scores


def donor_summary(scores):
    rows = []
    for subject, candidates in scores.groupby("target_id"):
        winners = candidates[candidates.other_similarity == candidates.other_similarity.max()]
        other_targets = scores[(scores.target_id != subject) & (scores.other_id != subject)]
        other_winners = other_targets.loc[other_targets.groupby("target_id").other_similarity.idxmax()]
        assert not other_targets.duplicated(["target_id","other_similarity"]).any()
        target_blind_r2 = candidates.set_index("other_id").loc[other_winners.other_id,"other_r2"].mean()
        rows.append(dict(target_id=subject, n_other=len(candidates), n_tied=len(winners),
                         selected_id="|".join(winners.other_id), selected_r2=winners.other_r2.mean(),
                         mean_other_r2=candidates.other_r2.mean(), own_r2=candidates.own_r2.mean(),
                         target_blind_nearest_donor_r2=target_blind_r2,
                         within_target_rho=spearmanr(candidates.other_similarity,candidates.other_r2).statistic))
    people = pd.DataFrame(rows)
    diff = (people.selected_r2 - people.mean_other_r2).to_numpy()
    ix = np.random.default_rng(20261120).integers(len(diff),size=(10000,len(diff)))
    targeted = (people.selected_r2 - people.target_blind_nearest_donor_r2).to_numpy()
    summary = dict(n_targets=len(diff), selected_mean_r2=people.selected_r2.mean(),
                   mean_other_r2=people.mean_other_r2.mean(), mean_selection_gain=diff.mean(),
                   gain_ci=np.quantile(diff[ix].mean(axis=1),[.025,.975]).tolist(),
                   target_blind_r2=people.target_blind_nearest_donor_r2.mean(),
                   target_aware_minus_blind=targeted.mean(),
                   target_aware_minus_blind_ci=np.quantile(targeted[ix].mean(axis=1),[.025,.975]).tolist(),
                   targets_improved=int((diff>0).sum()), unique_selected_donors=people.selected_id.nunique(),
                   mean_within_target_rank_correlation=people.within_target_rho.mean(),
                   n_tied_targets=int((people.n_tied>1).sum()))
    return people, summary


def verify_penalty_excluding_flag(graphs, ids, scores):
    # Retain the original simulation stream IDs when excluding a participant.
    mapping, validations, _ = target_data(graphs)
    losses = []
    for i, graph in enumerate(graphs[0]):
        training = pilot.simulate(pilot.reservoir(graph,transfer.GAIN),mapping,10000+i,0,pilot.FIT_N)
        model = pilot.RidgeFamily.from_design(training)
        losses.append([transfer.errors(model.predict(validations[i][0],alpha),validations[i])[0].mean()
                       for alpha in transfer.PENALTIES])
    loss = pd.DataFrame(losses,index=ids,columns=transfer.PENALTIES)
    for row in scores.itertuples():
        training_loss = loss.drop([FLAGGED,row.target_id,row.other_id])
        assert len(training_loss) == len(ids)-3 and training_loss.mean().idxmin() == row.relative_penalty


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    grid = pd.read_csv(OUT/"grid_pairs.csv",dtype={"target_id":str,"other_id":str})
    all_rows, selection, dense_selection = [], {}, {}
    for cohort in ("nki","bnu1"):
        graphs, ids, scores = prepare(cohort)
        people, selection[cohort] = donor_summary(scores)
        all_rows.append(people.assign(cohort=cohort))
        half = "odd" if cohort == "nki" else "all"
        primary = grid.query("cohort == @cohort and src_visit == 1 and target_visit == 2 and "
                             "src_half == @half and target_half == @half")
        dense = scores.merge(primary[["target_id","other_id","coarse_other","dense_own","dense_other"]],
                             on=["target_id","other_id"],validate="one_to_one")
        assert len(dense) == len(scores)
        np.testing.assert_allclose(dense.other_r2,dense.coarse_other,rtol=0,atol=1e-12)
        _, dense_selection[cohort] = donor_summary(dense.assign(other_r2=dense.dense_other,own_r2=dense.dense_own))
        if cohort != "nki":
            continue
        strict = scores[(scores.target_id != FLAGGED) & (scores.other_id != FLAGGED)]
        verify_penalty_excluding_flag(graphs,ids,strict)
        results = {}
        for name, subset in (("all",scores),("omit_target",scores[scores.target_id != FLAGGED]),
                             ("omit_target_donor_and_tuning",strict)):
            paired = subset.assign(similarity_gain=subset.own_similarity - subset.other_similarity)
            means = paired.groupby("target_id")[["gain","similarity_gain"]].mean()
            results[name] = dict(n_targets=len(means),n_pairs=len(paired),gain=means.gain.mean(),
                                 rho=spearmanr(means.gain,means.similarity_gain).statistic)
        (OUT/"flagged_sensitivity.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/"nearest_donor_people.csv",index=False)
    (OUT/"nearest_donor_summary.json").write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8")
    (OUT/"nearest_donor_dense_summary.json").write_text(json.dumps(dense_selection,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
