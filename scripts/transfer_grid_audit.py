"""Audit external transfer tuning by expanding and refining the ridge-penalty grid."""
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import json

from verify_results import SingularRidge, fixtures, join, stream
from verify_nki_calibration import adjacency
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "transfer-similarity"
WIDE = np.power(10., np.arange(-128, 49)/8)
ORIGINAL = np.logspace(-14, 4, 19)
ORIGINAL_INDEX = np.array([np.flatnonzero(WIDE == p).item() for p in ORIGINAL])
CONDITION = ["cohort", "src_visit", "src_half", "target_visit", "target_half"]


def transfer_loss_curve(model, validation):
    """Average validation MSE from target second moments over a fixed source ridge family."""
    x, y = validation
    a = (x - model.xbar) @ model.right
    y = y - model.ybar
    projected_y = model.projected_y
    factors = model.s / (model.s**2 + model.scale*WIDE[:,None])
    norm = y.size
    gram = (a.T @ a) * (projected_y @ projected_y.T) / norm
    cross = ((a.T @ y) * projected_y).sum(axis=1) / norm
    mse = np.mean(y*y) - 2*(factors @ cross) + ((factors @ gram) * factors).sum(axis=1)
    # The interval of transfer-relevant penalty strengths is also evaluated by direct predictions.
    check = np.array([np.flatnonzero(WIDE==p).item() for p in [.01,1.,10.,1e4]])
    raw_x, raw_y = validation
    direct = [np.mean((model.predict(raw_x, WIDE[k])-raw_y)**2) for k in check]
    np.testing.assert_allclose(mse[check], direct, atol=3e-11, rtol=3e-10)
    if not np.isfinite(mse).all() or mse.min() <= 0:
        raise ValueError("Invalid validation error in penalty sweep")
    return mse


def build_cells(weights, halves):
    mapping = np.zeros(weights.shape[-1])
    locations = stream(10).choice(len(mapping),8,replace=False)
    mapping[locations[:4]], mapping[locations[4:]] = .5,-.5
    cells = {}
    for ih, half in enumerate(halves):
        for iv in range(2):
            scans = []
            for index, matrix in enumerate(weights[ih,iv]):
                graph = adjacency(matrix)
                ident = 10000 + iv*weights.shape[2]+index
                source = SingularRidge(fixtures(graph,mapping,ident,0,2352))
                validation = fixtures(graph,mapping,ident,1,1176)
                test = join([fixtures(graph,mapping,ident,61,512,k) for k in range(4)])
                scans.append((source,validation,test))
            cells[(iv+1,half)] = scans
    return cells


def evaluate(source, target, people, condition):
    n = len(people)
    loss = np.array([transfer_loss_curve(s[0],t[1]) for s,t in zip(source,target)])
    @lru_cache(None)
    def test_r2(t, s, k):
        x, y = target[t][2]
        pred = source[s][0].predict(x,WIDE[k])
        return float((1-((pred-y)**2).mean(axis=0)/y.var(axis=0)).mean())

    pairs = []
    for t in range(n):
        for d in range(n):
            if t == d:
                continue
            others = (np.arange(n)!=t) & (np.arange(n)!=d)
            risk = loss[others].mean(axis=0)
            k = np.argmin(risk)
            old = ORIGINAL_INDEX[np.argmin(risk[ORIGINAL_INDEX])]
            assert risk[k] <= risk[old] + 1e-13
            pairs.append(dict(**condition, target_id=people[t],other_id=people[d],
                              coarse_alpha=WIDE[old],dense_alpha=WIDE[k],
                              coarse_own=test_r2(t,t,old),coarse_other=test_r2(t,d,old),
                              dense_own=test_r2(t,t,k),dense_other=test_r2(t,d,k)))
    return pairs, pd.DataFrame({"alpha":WIDE,"mean_donor_validation_mse":loss.mean(axis=0)}).assign(**condition)


def original_reference():
    nki = pd.read_csv(RESULTS/"nki-split-half/pair_scores.csv").assign(cohort="nki")
    beijing = pd.read_csv(RESULTS/"density-controls/frozen/single_donor_pairs.csv",
                          dtype={"target_id":str,"other_id":str})
    beijing = beijing[(beijing.cohort=="bnu1") & (beijing.source_input=="original_distinct_inputs")]
    beijing = beijing.rename(columns={"relative_penalty":"alpha"}).assign(
        src_visit=1,src_half="all",target_visit=2,target_half="all")
    wanted = CONDITION + ["target_id","other_id","alpha","own_r2","other_r2"]
    return pd.concat([nki[wanted],beijing[wanted]],ignore_index=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    records, curves = [], []
    for cohort, file in (("nki", ROOT/"data/nki/nki_reference.npz"),
                         ("bnu1", ROOT/"data/replication_candidates/corr/bnu1_dkt_retest.npz")):
        with np.load(file,allow_pickle=False) as data:
            people = data["subjects"].astype(str)
            halves = data["halves"].astype(str).tolist() if cohort=="nki" else ["all"]
            w = data["weights"] if cohort=="nki" else data["weights"][None]
        cells = build_cells(w,halves)
        comparisons = permutations(cells,2) if cohort=="nki" else [((1,"all"),(2,"all"))]
        for (sv,sh),(tv,th) in comparisons:
            condition = dict(zip(CONDITION,[cohort,sv,sh,tv,th]))
            rows, curve = evaluate(cells[sv,sh], cells[tv,th], people, condition)
            records.extend(rows)
            curves.append(curve)
    pairs = pd.DataFrame(records)
    for rule in ("coarse","dense"):
        pairs[f"{rule}_gain"] = pairs[f"{rule}_own"] - pairs[f"{rule}_other"]
    original = original_reference()
    checked=pairs.merge(original,on=CONDITION+["target_id","other_id"],validate="one_to_one")
    assert len(checked) == len(pairs) == len(original) == 10392
    np.testing.assert_allclose(checked.coarse_alpha,checked.alpha,rtol=1e-12,atol=0)
    error = float(max(abs(checked.coarse_own-checked.own_r2).max(),
                      abs(checked.coarse_other-checked.other_r2).max()))
    assert error < 1e-7
    subjects = pairs.groupby(CONDITION+["target_id"],as_index=False)[
        [f"{r}_{m}" for r in ("coarse","dense") for m in ("own","other","gain")]].mean()
    available = pairs.groupby(CONDITION)
    summary = []
    for key, sub in subjects.groupby(CONDITION):
        draw = np.random.default_rng(20261120).integers(len(sub),size=(10000,len(sub)))
        row = dict(zip(CONDITION,key))
        for field in sub.columns[6:]:
            values = sub[field].to_numpy()
            row[field] = values.mean()
            row[field+"_lo"],row[field+"_hi"] = np.quantile(values[draw].mean(axis=1),[.025,.975])
        for field in ("own","gain"):
            delta = (sub[f"dense_{field}"]-sub[f"coarse_{field}"]).to_numpy()
            row[f"{field}_change"] = delta.mean()
            row[f"{field}_change_lo"],row[f"{field}_change_hi"]=np.quantile(delta[draw].mean(axis=1),[.025,.975])
        raw = available.get_group(key)
        for field in ("coarse_alpha","dense_alpha"):
            row[field+"_min"] = raw[field].min()
            row[field+"_max"] = raw[field].max()
            row[field+"_mode"] = raw[field].mode().iloc[0]
        row["dense_wins"] = int((sub.dense_gain>0).sum())
        row["n"] = len(sub)
        summary.append(row)
    pd.concat(curves).to_csv(OUT/"grid_validation.csv",index=False)
    pairs.to_csv(OUT/"grid_pairs.csv",index=False)
    subjects.to_csv(OUT/"grid_people.csv",index=False)
    frame = pd.DataFrame(summary)
    frame.to_csv(OUT/"grid_summary.csv",index=False)
    metadata = dict(original_grid=ORIGINAL.tolist(), expanded_grid=WIDE.tolist(),
                    original_pairs_reproduced=len(checked), max_abs_replay_score_error=error,
                    tuning="own-source/own-target donor validation curves excluding both compared identities",
                    uncertainty="10000 participant bootstrap draws, fixed training bank and tuning curves",
                    validation_metric="unweighted MSE across ten delays and 1176 donor validation steps",
                    test_metric="average ten held-out R2 scores over 2048 fresh simulated test steps")
    (OUT/"grid_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(frame[CONDITION+["coarse_own","dense_own","coarse_gain","dense_gain","dense_alpha_min",
                           "dense_alpha_max"]].round(5).to_string(index=False))


if __name__ == "__main__":
    run()
