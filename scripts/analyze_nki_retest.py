"""Compare visit-specific frozen decoders with participant-level uncertainty."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t


OUT = Path(__file__).resolve().parents[1] / "results" / "nki-retest"
KEYS = ["half", "source_visit", "target_visit"]
NDRAWS = 49_999


def advantage(errors):
    n = len(errors)
    own = np.diag(errors)
    return (errors.sum(axis=0) - own) / (n - 1) - own


def inference(errors):
    """Drop or permute complete identities in the donor by target matrix."""
    n = len(errors)
    effect = advantage(errors).mean()
    leave_one = []
    positions = np.arange(n)
    for i in positions:
        keep = positions != i
        leave_one.append(advantage(errors[np.ix_(keep, keep)]).mean())
    se = np.std(leave_one, ddof=0) * np.sqrt(n - 1)
    width = t.ppf(.975, n - 1) * se

    # For fixed row/column margins, larger advantage is equivalent to a
    # smaller assigned trace. Random perfect matchings retain donor dependence.
    obs = np.diag(errors).mean()
    rng = np.random.default_rng(20261120)
    ranks = rankdata(errors, axis=0, method="average")
    observed_rank = np.diag(ranks).mean()
    count = 0
    rank_count = 0
    for start in range(0, NDRAWS, 1000):
        size = min(1000, NDRAWS - start)
        perms = rng.random((size, n)).argsort(axis=1)
        count += np.count_nonzero(errors[perms, positions].mean(axis=1) <= obs)
        rank_count += np.count_nonzero(ranks[perms, positions].mean(axis=1) <= observed_rank)
    return dict(advantage_ci_low=effect-width, advantage_ci_high=effect+width,
                permutation_p=(count+1)/(NDRAWS+1),
                secondary_rank_permutation_p=(rank_count+1)/(NDRAWS+1), permutations=NDRAWS)


def analyze(pairs, pooled, primary_inference=True):
    subjects = sorted(pairs.source_subject.unique())
    assert subjects == sorted(pairs.target_subject.unique())
    n = len(subjects)
    assert n == 25 and len(pairs) == n*n*10
    assert set(pairs.delay)==set(range(1, 11))
    assert not pairs.duplicated(["source_subject", "target_subject", "delay"]).any()
    assert len(pooled)==n*10 and not pooled.duplicated(["target_subject", "delay"]).any()
    grouped = pairs.groupby(["source_subject","target_subject"]).mean(numeric_only=True)
    def matrix(metric):
        x = grouped[metric].unstack("target_subject").loc[subjects, subjects].to_numpy()
        assert np.isfinite(x).all()
        return x
    errors, r2, wiring = map(matrix, ["normalized_mse","r2","wiring_correlation"])
    pooled = pooled.groupby("target_subject").mean(numeric_only=True).loc[subjects]
    baseline = matrix("target_own_nmse")
    base_r2 = matrix("target_own_r2")
    np.testing.assert_allclose(baseline, np.repeat(baseline[:1],n,axis=0))
    np.testing.assert_allclose(base_r2, np.repeat(base_r2[:1],n,axis=0))
    positive = advantage(errors)
    rows = []
    for j, name in enumerate(subjects):
        others = np.flatnonzero(np.arange(n) != j)
        ordered = others[np.argsort(wiring[others, j],kind="stable")]
        corrs, own_corr = wiring[ordered,j], wiring[j,j]
        err, own = errors[:,j], errors[j,j]
        best = ordered[-1]
        support = bool(corrs[0] <= own_corr <= corrs[-1])
        # Average errors for identical correlation values before interpolation.
        xp, mapping = np.unique(corrs,return_inverse=True)
        fp = np.bincount(mapping, weights=err[ordered])/np.bincount(mapping)
        interp = np.interp(own_corr,xp,fp) if support else np.nan
        decoder_rank = 1 + (err[others] < own).sum() + .5*(err[others] == own).sum()
        wiring_rank = 1 + (corrs > own_corr).sum() + .5*(corrs == own_corr).sum()
        rows.append(dict(subject=name, within_scan_nmse=baseline[0,j], within_scan_r2=base_r2[0,j],
                         self_nmse=own,self_r2=r2[j,j],other_mean_nmse=err[others].mean(),
                         other_median_nmse=np.median(err[others]),other_mean_r2=r2[others,j].mean(),
                         advantage_nmse=positive[j],advantage_r2=r2[j,j]-r2[others,j].mean(),
                         best_wiring_other=subjects[best],best_wiring_other_nmse=err[best],
                         best_wiring_other_r2=r2[best,j],self_wiring_corr=own_corr,
                         best_other_wiring_corr=corrs[-1],worst_other_wiring_corr=corrs[0],
                         decoder_rank=decoder_rank,wiring_rank=wiring_rank,
                         fraction_other_decoders_beaten=(n-decoder_rank)/(n-1),
                         within_similarity_support=support,matched_advantage_nmse=interp-own,
                         pooled_nmse=pooled.normalized_mse.iloc[j],pooled_r2=pooled.r2.iloc[j]))
    a = pd.DataFrame(rows)
    summary = {c: getattr(a,c).mean() for c in ["self_nmse","self_r2","other_mean_nmse","other_mean_r2",
              "advantage_nmse","advantage_r2","pooled_nmse","pooled_r2","within_scan_nmse",
              "within_scan_r2","best_wiring_other_nmse","best_wiring_other_r2","fraction_other_decoders_beaten"]}
    supported = a[a.within_similarity_support]
    summary.update(people=n,median_advantage=a.advantage_nmse.median(),
                   n_better_than_mean_other=int((a.advantage_nmse>0).sum()),
                   n_better_than_median_other=int((a.self_nmse < a.other_median_nmse).sum()),
                   n_better_than_pooled=int((a.self_nmse < a.pooled_nmse).sum()),
                   n_better_than_nearest_wiring=int((a.self_nmse < a.best_wiring_other_nmse).sum()),
                   n_better_than_constant=int((a.self_r2>0).sum()),
                   median_self_nmse=a.self_nmse.median(),median_self_r2=a.self_r2.median(),
                   median_pooled_r2=a.pooled_r2.median(), median_decoder_rank=a.decoder_rank.median(),
                   n_decoder_top1=int((a.decoder_rank==1).sum()),n_wiring_top1=int((a.wiring_rank==1).sum()),
                   n_wiring_supported=len(supported),
                   matched_advantage_nmse=supported.matched_advantage_nmse.mean() if len(supported)>=10 else np.nan,
                   max_single_pair_nmse=errors.max())
    if primary_inference:
        summary.update(inference(errors))
    return a, summary


def main():
    labels={"source_subject":str,"target_subject":str}
    all_pairs = pd.read_csv(OUT/"pair_scores.csv",dtype=labels)
    pool = pd.read_csv(OUT/"pooled_scores.csv",dtype={"target_subject":str})
    rows, subjects, lags = [], [], []
    for keys, frame in all_pairs.groupby(KEYS):
        context = dict(zip(KEYS,keys))
        p = pool[(pool.half==keys[0])&(pool.source_visit==keys[1])&(pool.target_visit==keys[2])]
        results, summary = analyze(frame,p)
        rows.append({**context,**summary})
        subjects.append(results.assign(**context))
        for delay, f in frame.groupby("delay"):
            own = f[f.source_subject==f.target_subject]
            other = f[f.source_subject!=f.target_subject]
            p_lag = p[p.delay==delay]
            row = dict(**context, delay=delay, self_r2=own.r2.mean(),other_r2=other.r2.mean(),
                       pooled_r2=p_lag.r2.mean(),within_scan_r2=own.target_own_r2.mean(),
                       self_nmse=own.normalized_mse.mean(),other_nmse=other.normalized_mse.mean(),
                       pooled_nmse=p_lag.normalized_mse.mean(),within_scan_nmse=own.target_own_nmse.mean())
            lags.append(row)
    assert len(rows)==4 and len(lags)==40
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)
    pd.concat(subjects).to_csv(OUT/"participant_summary.csv",index=False)
    pd.DataFrame(lags).to_csv(OUT/"delay_summary.csv",index=False)


if __name__ == "__main__":
    main()
