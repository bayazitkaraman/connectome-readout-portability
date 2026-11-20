"""Independently trace the NKI split matrices and reconstruct their scores and anatomical matches."""
import hashlib
import json
from pathlib import Path

from verify_results import SingularRidge, fixtures, join, stream
from verify_nki_calibration import GRID, adjacency
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/nki"
OUT = ROOT / "results/nki-split-half"
FLAG = "A00063383"
CONDITION = ["src_visit", "src_half", "target_visit", "target_half"]


def file_hash(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source():
    hdf = DATA / "inputs.h5"
    assert file_hash(hdf, "md5") == "f457d4a860d79654672143c1ace95ee1"
    assert file_hash(DATA / "crosswalk.csv", "sha256") == "f9181517bac7dc899dc5aeeda050158396dbcbecc1fd2c04716fa5ae987cdd25"
    table = pd.read_hdf(hdf, key="graphs")
    selected = table[(table.simulation == "ref") & (table.pipeline == "det")]
    assert len(selected) == 100
    crosswalk = pd.read_csv(DATA / "crosswalk.csv", dtype=str)
    mapped = crosswalk.groupby(["A_number","old_session_name"]).new_session_id.unique().to_dict()
    raw = np.load(DATA / "nki_reference.npz", allow_pickle=False)
    graphs = raw["weights"]
    ids = raw["subjects"].astype(str).tolist()
    labels = list(raw["halves"].astype(str))
    assert labels == ["odd","even"]
    assert len(ids) == 25 and graphs.shape == (2,2,25,83,83)
    assert not selected.duplicated(["subject","session","directions"]).any()
    for row in selected.itertuples():
        i, h = ids.index(row.subject), labels.index(row.directions)
        visit = np.flatnonzero(raw["session_labels"][:, i] == row.session).item()
        np.testing.assert_array_equal(graphs[h,visit,i], row.graph)
        key = (row.subject, row.session)
        standard = raw["standard_session_labels"][visit,i]
        if key in mapped:
            np.testing.assert_array_equal(mapped[key], [standard])
        else:
            assert key == ("A00034854","NFB2R") and standard == "unmapped"
    flag = ids.index(FLAG)
    assert raw["session_labels"][:,flag].tolist() == ["CLGA","CLG2R"]
    assert raw["standard_session_labels"][:,flag].tolist() == ["BAS1","TRT"]
    return ids, labels, graphs


def verify():
    ids, halves, graphs = verify_source()
    n = len(ids)
    pair = pd.read_csv(OUT/"pair_scores.csv",dtype={"target_id":str,"other_id":str})
    person = pd.read_csv(OUT/"subject_scores.csv",dtype={"target_id":str})
    assert not pair.duplicated(CONDITION+["target_id","other_id"]).any()
    assert not person.duplicated(CONDITION+["target_id"]).any()
    a, b = np.triu_indices(graphs.shape[-1],1)
    edges = graphs[:,:,:,a,b]
    inputs = np.zeros(graphs.shape[-1])
    chosen = stream(10).choice(len(inputs),8,replace=False)
    inputs[chosen[:4]], inputs[chosen[4:]] = .5, -.5
    models, validation, testing = {}, {}, {}
    active = (0, n//2, ids.index(FLAG))
    for h in range(2):
        for v in range(2):
            for i, identity in enumerate(ids):
                ident = 10000 + n*v + i
                graph = adjacency(graphs[h,v,i])
                models[h,v,i] = SingularRidge(fixtures(graph,inputs,ident,0,2352))
                validation[h,v,i] = fixtures(graph,inputs,ident,1,1176)
                if i in active:
                    testing[h,v,i] = join([fixtures(graph,inputs,ident,61,512,block) for block in range(4)])

    checks, max_error, max_reference_error = [], 0., 0.
    for condition, block in pair.groupby(CONDITION):
        srcv, srch, tgtv, tgth = condition
        vs, vt = srcv-1, tgtv-1
        hs, ht = halves.index(srch), halves.index(tgth)
        assert len(block)==n*(n-1) and (block.target_id != block.other_id).all()
        # The first n rows index target scans, the second n source scans.
        correlations = np.corrcoef(edges[ht,vt], edges[hs,vs])[:n,n:]
        t = np.array([ids.index(name) for name in block.target_id])
        d = np.array([ids.index(name) for name in block.other_id])
        own, other = correlations[t,t], correlations[t,d]
        np.testing.assert_allclose(block.own_similarity, own, atol=2e-13)
        np.testing.assert_allclose(block.other_similarity, other, atol=2e-13)
        np.testing.assert_allclose(block.anatomy_advantage, own-other, atol=4e-13)

        donor_loss = []
        for i in range(n):
            fit = models[hs,vs,i]
            x,y = validation[ht,vt,i]
            donor_loss.append([np.mean((fit.predict(x,alpha)-y)**2) for alpha in GRID])
        donor_loss = np.asarray(donor_loss)
        # Each validation curve comes from another person's matched scans.
        alpha = [GRID[np.argmin(donor_loss[np.setdiff1d(np.arange(n),[i,j])].mean(axis=0))]
                 for i,j in zip(t,d)]
        np.testing.assert_allclose(block.alpha,alpha,rtol=1e-10,atol=0)
        selected_count = 0
        for r in block.itertuples():
            i,j = ids.index(r.target_id),ids.index(r.other_id)
            if i not in active or j not in {0,3,12,24}:
                continue
            x,y = testing[ht,vt,i]
            vx = y.var(axis=0)
            estimates = []
            for owner in [i,j]:
                prediction = models[hs,vs,owner].predict(x,r.alpha)
                estimates.append(float((1-((prediction-y)**2).mean(axis=0)/vx).mean()))
            recorded = np.array([r.own_r2,r.other_r2])
            np.testing.assert_allclose(estimates, recorded, atol=1e-7, rtol=2e-7)
            max_error = max(max_error,float(abs(recorded-estimates).max()))
            selected_count += 1
        rank = np.array([1+np.sum(row>row[i])+.5*(np.sum(row==row[i])-1)
                         for i,row in enumerate(correlations)])
        p = person.set_index(CONDITION).loc[condition].set_index("target_id").loc[ids]
        np.testing.assert_array_equal(p.anatomic_rank, rank)
        np.testing.assert_array_equal(p.anatomic_top1, rank==1)
        for i in active:
            within_model = models[ht,vt,i]
            alpha = within_model.tune(validation[ht,vt,i], GRID)
            x, y = testing[ht,vt,i]
            own_target_r2 = (1-((within_model.predict(x,alpha)-y)**2).mean(axis=0)/y.var(axis=0)).mean()
            recorded_reference = p.loc[ids[i],"reference_r2"]
            np.testing.assert_allclose(own_target_r2, recorded_reference, rtol=1e-6, atol=1e-8)
            max_reference_error = max(max_reference_error, float(abs(own_target_r2 - recorded_reference)))
        checks.append(dict(source_visit=int(srcv), source_half=srch, target_visit=int(tgtv),target_half=tgth,
                           anatomical_ranks_checked=n,penalties_checked=len(block),
                           direct_decoder_score_pairs=selected_count,anatomical_top1=int((rank==1).sum())))
    assert len(checks) == 12
    i = ids.index(FLAG)
    anomaly = edges[:,:,i].reshape(4,-1)
    triplets = pd.read_csv(ROOT / "results/anatomy-advantage/participants.csv", dtype={"subject":str})
    weak = triplets.query("cohort == 'nki' and mix_rule == 'equal_donor' and anatomy_advantage < 0")
    raw = np.load(DATA / "nki_reference.npz", allow_pickle=False)
    v2_labels = raw["standard_session_labels"][1, [ids.index(target) for target in weak.subject]]
    counts = pd.Series(v2_labels).value_counts().to_dict()
    assert counts == {"BAS2":5, "FLU1":2, "TRT":3}
    report = dict(raw_hdf_graphs_checked=100, crosswalk_unmapped_codes=1, condition_audits=checks,
                  max_direct_score_error=max_error, max_direct_within_target_reference_error=max_reference_error,
                  three_control_weak_second_visits=counts, flagged_participant=FLAG,
                  flagged_legacy_visit_codes=["CLGA","CLG2R"], flagged_standard_visit_codes=["BAS1","TRT"],
                  flagged_graph_order=["baseline_odd","repeat_odd","baseline_even","repeat_even"],
                  flagged_graph_correlations=np.corrcoef(anomaly).tolist(),
                  flagged_edge_counts=np.count_nonzero(anomaly,axis=1).tolist())
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"conditions":len(checks),"all_penalties":sum(r["penalties_checked"] for r in checks),
                      "direct_score_pairs":sum(r["direct_decoder_score_pairs"] for r in checks),
                      "max_error":max_error}))


if __name__ == "__main__":
    verify()
