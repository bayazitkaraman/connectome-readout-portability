"""How much edge-weight correlation is expected from shared support with randomized positive weights?"""
from pathlib import Path

from paired_pool_augmentation import read_graphs, ROOT
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUT = ROOT / "results/density-controls/mechanisms"


def support_expectation(x, y):
    """Exact mean Pearson r after independent positive-weight permutations within the two masks."""
    u, v = x > 0, y > 0
    mx, my = x.mean(), y.mean()
    exy = (u & v).sum() / x.size * x[u].mean() * y[v].mean()
    return (exy - mx*my) / (x.std() * y.std())


def permutation_check(x, y):
    rng = np.random.default_rng(201119)
    i = np.flatnonzero(x)
    j = np.flatnonzero(y)
    denominator = (x.std()*y.std()) * len(x)
    mean_term = len(x)*x.mean()*y.mean()
    p, q = np.zeros_like(x), np.zeros_like(y)
    rho = []
    for _ in range(1600):
        p[i] = rng.permutation(x[i])
        q[j] = rng.permutation(y[j])
        rho.append((p@q-mean_term)/denominator)
    se = np.std(rho, ddof=1) / np.sqrt(len(rho))
    assert abs(np.mean(rho)-support_expectation(x,y)) <= 5 * se + 1e-10


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(ROOT/"results/anatomy-advantage/pairs.csv", dtype={"subject":str,"other_subject":str})
    rows = []
    for cohort in ("nki","bnu1"):
        graphs, subjects, _ = read_graphs(cohort)
        a, b = np.triu_indices(graphs.shape[-1],1)
        edges = graphs[:,:,a,b].astype(float)
        lookup = {s:i for i,s in enumerate(subjects)}
        permutation_check(edges[1,0], edges[0,0])
        selection = matches[(matches.cohort == cohort) & (matches.mix_rule == "equal_donor")]
        for pair in selection.itertuples():
            i, j = lookup[pair.subject], lookup[pair.other_subject]
            target, own, other = edges[1,i], edges[0,i], edges[0,j]
            np.testing.assert_allclose(np.corrcoef(target,own)[0,1],pair.own_similarity,atol=2e-13)
            np.testing.assert_allclose(np.corrcoef(target,other)[0,1],pair.other_similarity,atol=2e-13)
            own_null = support_expectation(target,own)
            other_null = support_expectation(target,other)
            rows.append(dict(cohort=cohort,subject=pair.subject,other_subject=pair.other_subject,
                             own_r=pair.own_similarity,other_r=pair.other_similarity,
                             own_expected_r=own_null,other_expected_r=other_null,
                             observed_difference=pair.anatomy_advantage,expected_difference=own_null-other_null,
                             accuracy_advantage=pair.accuracy_advantage))
    joined=pd.DataFrame(rows)
    joined.to_csv(OUT/"support_weight_permutation_pairs.csv",index=False)
    summary=[]
    for cohort, frame in joined.groupby("cohort"):
        by_person=frame.groupby("subject").mean(numeric_only=True)
        mean=by_person.mean().to_dict()
        mean.update(cohort=cohort,n=len(by_person),rank_raw_vs_mask_expectation=
                    spearmanr(by_person.observed_difference,by_person.expected_difference).statistic)
        summary.append(mean)
    pd.DataFrame(summary).to_csv(OUT/"support_weight_permutation_means.csv",index=False)


if __name__ == "__main__":
    run()
