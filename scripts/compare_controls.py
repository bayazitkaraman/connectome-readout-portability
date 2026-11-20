"""Test whether a frozen readout is more portable on real than surrogate wiring."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_feasibility as pilot
import numpy as np
import pandas as pd
from profile_null import build_strata, make_profile_null

OUT = ROOT / "results" / "control-comparison"
ATLAS = ROOT / "data" / "atlas"
PENALTIES = np.logspace(-14, 4, 19)
FAMILIES = ("roles", "structure", "spatial")
PARTIAL = "spatial_partial"
BUDGETED = "spatial_budgeted"
REPEATS = 24
GAIN = .9


def upper_tags(a, b):
    return 3 * np.minimum(a, b) + np.maximum(a, b)


def groups(adjs, template_ids, family, roles, distance, hemisphere):
    """Fixed design parameters are inferred exclusively in the inner training set."""
    i, j = np.triu_indices(adjs.shape[1], 1)
    features = [upper_tags(roles[i], roles[j])]
    if family in ("structure", "spatial", PARTIAL, BUDGETED):
        strata = build_strata(adjs[template_ids], mean_weight_bins=3,
                              support_bins=3, endpoint_strength_bins=3)
        features.append(strata)
    if family in ("spatial", PARTIAL, BUDGETED):
        if distance is None or hemisphere is None:
            raise RuntimeError("The spatial family requires a verified atlas.")
        d = distance[i, j]
        features.append(np.searchsorted(np.quantile(d, [.25,.5,.75]), d, side="right"))
        features.append(upper_tags(hemisphere[i], hemisphere[j]))
    return np.unique(np.column_stack(features), axis=0, return_inverse=True)[1].ravel()


def validation_rows(net, mapping, person, n=None):
    data = pilot.simulate(net, mapping, person+1, 1, pilot.VALID_N)
    return data if n is None else pilot.sampled(data, n)


def score_population(graphs, mapping, train, inner, val, test):
    """Same fits as run_feasibility.evaluate, skipping irrelevant simulations."""
    inner_set = set(inner)
    inner_stats, train_stats, held_validation = [], [], []
    for p in train:
        net = pilot.reservoir(graphs[p], GAIN)
        data = pilot.simulate(net, mapping, p+1, 0, pilot.FIT_N)
        train_stats.append(pilot.Moment.from_data(pilot.sampled(data, 42)))
        if p in inner_set:
            inner_stats.append(pilot.Moment.from_data(pilot.sampled(data, 56)))
        else:
            held_validation.append(validation_rows(net, mapping, p, 84))
    assert set(train) - inner_set == set(val) and len(held_validation)==14
    alpha = pilot.RidgeFamily(pilot.merge(inner_stats)).tune(
        pilot.combine(held_validation), PENALTIES)
    shared = pilot.RidgeFamily(pilot.merge(train_stats))
    for p in test:
        net = pilot.reservoir(graphs[p], GAIN)
        own = pilot.RidgeFamily.from_design(pilot.simulate(net, mapping, p+1, 0, pilot.FIT_N))
        a = own.tune(validation_rows(net, mapping, p), PENALTIES)
        x, y = pilot.combine([pilot.simulate(net, mapping, p+1, 2, pilot.TEST_N, rep)
                              for rep in range(pilot.TEST_STREAMS)])
        v = y.var(axis=0)
        se = np.square(shared.predict(x, alpha)-y).mean(axis=0)
        oe = np.square(own.predict(x, a)-y).mean(axis=0)
        yield p, 1 - se/v, 1 - oe/v, (se-oe)/pilot.SIGNAL_VAR


def read_atlas():
    dist_path = ATLAS / "reference_eu_distance.csv"
    hemi_path = ATLAS / "hemispheres.csv"
    if not dist_path.exists() or not hemi_path.exists():
        return None, None
    d = np.loadtxt(dist_path, delimiter=",")
    table = pd.read_csv(hemi_path)
    hemi = np.unique(table["hemisphere"], return_inverse=True)[1]
    assert d.shape == (68,68) and len(hemi) == 68 and len(np.unique(hemi)) == 2
    assert np.allclose(d,d.T) and np.isfinite(d).all()
    return d, hemi


def quality(raw, fake, test, distance):
    n = raw.shape[1]
    tri = np.triu_indices(n, 1)
    e0, e1 = raw[:,tri[0],tri[1]], fake[:,tri[0],tri[1]]
    c0, c1 = np.corrcoef(e0), np.corrcoef(e1)
    pair_error = float(abs(c0-c1).max())
    assert pair_error < 1e-12
    np.testing.assert_array_equal(np.sort(e0,axis=1),np.sort(e1,axis=1))
    g0,g1 = raw[test],fake[test]
    n0,n1=(g0>0).sum(axis=2),(g1>0).sum(axis=2)
    s0,s1=g0.sum(axis=2),g1.sum(axis=2)
    ans = dict(max_pair_correlation_error=pair_error,
               relative_degree_shift=float((abs(n0-n1).sum(axis=1)/n0.sum(axis=1)).mean()),
               relative_strength_shift=float((abs(s0-s1).sum(axis=1)/s0.sum(axis=1)).mean()),
               fraction_weight_redistributed=float((abs(g0-g1).sum(axis=(1,2)) /
                                                    (2*g0.sum(axis=(1,2)))).mean()),
               fraction_active_weights_changed=float(
                   ((g0!=g1)&(g0>0)).sum()/(g0>0).sum()))
    if distance is not None:
        m0 = (g0 * distance).sum(axis=(1,2))/s0.sum(axis=1)
        m1 = (g1 * distance).sum(axis=(1,2))/s1.sum(axis=1)
        ans["relative_mean_weighted_distance_shift"] = float((abs(m0-m1)/m0).mean())
    return ans


def run_draw(job):
    family, draw = job
    template = pilot.load_adjacencies()
    mapping = pilot.input_map(template.shape[1])
    roles = np.sign(mapping[0][0]).astype(int) + 1
    distance, hemi = read_atlas()
    outcomes, diagnostics = [], []
    for fold, train, inner, val, test in pilot.folds(len(template)):
        keys = groups(template, inner, family, roles, distance, hemi)
        # Reuse the seed across folds of a draw; training-based strata can vary by fold.
        seed = np.random.SeedSequence([pilot.SEED,48,draw])
        if family=="identity":
            fake = template
        elif family == BUDGETED:
            from budgeted_null import budgeted_null
            fake = budgeted_null(template, inner, keys, seed)
        else:
            fake = make_profile_null(template, keys, seed, partial_pairs=(family == PARTIAL))
        diagnostics.append(dict(family=family,draw=draw,fold=fold,classes=len(set(keys)),
                                **quality(template,fake,test,distance)))
        for p, sr2, ir2, cost in score_population(fake,mapping,train,inner,val,test):
            for k, s, i, c in zip(pilot.DELAYS,sr2,ir2,cost):
                outcomes.append((family,draw,fold,p+1,int(k),s,i,c))
    df = pd.DataFrame(outcomes,columns=["family","draw","fold","participant","delay",
                                        "shared_r2","own_r2","cost_nmse"])
    return df, diagnostics


def existing_real():
    old = pd.read_csv(pilot.OUT/"sensitivity_participant_scores.csv")
    old = old[(old.map_key==10)&old.decoder.isin(["shared","individual"])]
    w = old.pivot(index=["fold","participant","delay"],columns="decoder",
                  values=["r2","normalized_mse"])
    out = pd.DataFrame(dict(shared_r2=w["r2","shared"],own_r2=w["r2","individual"],
                            cost_nmse=w["normalized_mse","shared"]-w["normalized_mse","individual"]))
    return out.reset_index()


def aggregate(data, diagnostics):
    all_summary = []
    real = existing_real()
    observed = real.groupby("delay").mean(numeric_only=True)
    for family in sorted(data.family.unique()):
        samples = data[data.family==family].copy()
        merged = samples.merge(real,on=["fold","participant","delay"],
                               suffixes=("","_real"),validate="many_to_one")
        merged["contrast"] = merged.cost_nmse-merged.cost_nmse_real
        cost = merged.groupby(["draw","participant"]).contrast.mean().unstack("participant")
        assert cost.shape==(REPEATS,70)
        # Resample paired participant IDs and draws; fitted folds held fixed.
        random = pilot.rng(49)
        people = random.multinomial(70,np.full(70,1/70),size=10000)
        weights = random.multinomial(REPEATS,np.full(REPEATS,1/REPEATS),size=10000)
        means = ((weights @ cost.to_numpy()) * people).sum(axis=1)/(70*REPEATS)
        intervals = np.quantile(means,[.025,.975])
        quality_checks = diagnostics[diagnostics.family==family]
        deltas = samples.groupby("delay").own_r2.mean() - observed.own_r2
        own_shift = float(deltas.mean())
        rms = float(np.sqrt((deltas*deltas).mean()))
        deg = float(quality_checks.relative_degree_shift.mean())
        strength = float(quality_checks.relative_strength_shift.mean())
        row = dict(family=family,draws=REPEATS,people=70,shared_r2=samples.shared_r2.mean(),
                   own_r2=samples.own_r2.mean(),cost_nmse=samples.cost_nmse.mean(),
                   observed_shared_r2=real.shared_r2.mean(),observed_own_r2=real.own_r2.mean(),
                   observed_cost_nmse=real.cost_nmse.mean(),null_minus_human=float(cost.to_numpy().mean()),
                   ci_low=intervals[0],ci_high=intervals[1],
                   min_draw_contrast=cost.mean(axis=1).min(),max_draw_contrast=cost.mean(axis=1).max(),
                   mean_own_r2_shift=own_shift,rms_delay_own_r2_shift=rms,
                   relative_degree_shift=deg,relative_strength_shift=strength,
                   fraction_active_weights_changed=quality_checks.fraction_active_weights_changed.mean(),
                   max_pair_correlation_error=quality_checks.max_pair_correlation_error.max(),
                   own_recall_matches=abs(own_shift)<=.03 and rms<=.05,
                   degree_strength_matches=deg<=.20 and strength<=.20)
        if "relative_mean_weighted_distance_shift" in quality_checks:
            row["relative_mean_weighted_distance_shift"]=quality_checks.relative_mean_weighted_distance_shift.mean()
        if "fraction_weight_redistributed" in quality_checks:
            row["fraction_weight_redistributed"]=quality_checks.fraction_weight_redistributed.mean()
        all_summary.append(row)
    return pd.DataFrame(all_summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families",nargs="+",choices=(*FAMILIES, PARTIAL, BUDGETED),default=list(FAMILIES))
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--check",action="store_true",help="Verify identity against the original run.")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.check:
        new,_=run_draw(("identity",0))
        compare=new.merge(existing_real(),on=["fold","participant","delay"],suffixes=("","_old"))
        assert len(compare)==700
        for c in ("shared_r2","own_r2","cost_nmse"):
            np.testing.assert_allclose(compare[c],compare[c+"_old"],rtol=0,atol=1e-9)
        print("Identity transformation reproduced all 700 paired scores.",flush=True)
        return
    for family in args.families:
        start = time.monotonic()
        batch=[]; qc=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for idx, (scores, check) in enumerate(pool.map(run_draw,[(family,i) for i in range(REPEATS)]),1):
                batch.append(scores)
                qc.extend(check)
                if idx%6==0:
                    print(f"{family}: completed {idx}/{REPEATS} cohorts",flush=True)
        df=pd.concat(batch,ignore_index=True)
        dq=pd.DataFrame(qc)
        assert len(df)==REPEATS*70*10 and len(dq)==REPEATS*5
        df.to_csv(OUT/f"{family}_scores.csv",index=False)
        dq.to_csv(OUT/f"{family}_quality.csv",index=False)
        aggregate(df,dq).to_csv(OUT/f"{family}_summary.csv",index=False)
        print(f"Saved {family} after {time.monotonic()-start:.1f} s",flush=True)


if __name__=="__main__":
    main()

