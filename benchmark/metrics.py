"""Metrics for matrices with source participants in rows and targets in columns."""
import numpy as np
from scipy.stats import rankdata

TIE_ATOL = 1.e-12


def wiring_similarity(source, target):
    i, j = np.triu_indices(source.shape[-1], k=1)

    def standardize(graphs):
        edges = graphs[:, i, j].astype(float, copy=True)
        if (np.ptp(edges, axis=1) == 0).any():
            raise ValueError("Edge-weight Pearson correlation is undefined for a constant graph.")
        edges /= np.max(np.abs(edges), axis=1, keepdims=True)
        edges -= edges.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(edges, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("Edge-weight Pearson correlation is undefined for a constant graph.")
        return edges / norm

    return np.clip(standardize(source) @ standardize(target).T, -1, 1)


def identify(scores):
    """Midrank of own source, and top-one probability under uniform tie breaking."""
    own = np.diag(scores)
    difference = scores - own
    n_above = np.sum(difference > TIE_ATOL, axis=0)
    n_equal = np.sum(np.abs(difference) <= TIE_ATOL, axis=0)
    return 1 + n_above + (n_equal - 1) / 2, (n_above == 0) / n_equal


def reliability(first, second):
    x = np.column_stack((first, second))
    n = len(x)
    people = x.mean(axis=1, keepdims=True)
    visits = x.mean(axis=0, keepdims=True)
    between = 2 * np.var(people, ddof=1)
    visit = n * np.var(visits, ddof=1)
    residual = np.square(x - people - visits + x.mean()).sum() / (n - 1)
    denom = between + residual + 2 * (visit - residual) / n
    ranks = np.apply_along_axis(rankdata, 0, x)
    rho = np.corrcoef(ranks.T)[0, 1] if np.all(np.ptp(ranks, axis=0) > 0) else np.nan
    return {"spearman": rho, "icc_absolute_single": (between - residual) / denom if denom else np.nan}


def bin_curve(wiring, scores, edges):
    own = np.eye(len(scores), dtype=bool)
    index = np.searchsorted(edges, wiring, side="right")
    limits = [-1, *edges, 1]
    rows = []
    for i, (low, high) in enumerate(zip(limits, limits[1:])):
        for name, mask in (("own", own), ("other", ~own)):
            use = (index == i) & mask
            rows.append(dict(source_type=name, low=low, high=high, n_pairs=int(use.sum()),
                             n_targets=int(use.any(axis=0).sum()),
                             mean_similarity=wiring[use].mean() if use.any() else np.nan,
                             mean_r2=scores[use].mean() if use.any() else np.nan,
                             fraction_r2_at_least_010=(scores[use]>=.10).mean() if use.any() else np.nan,
                             fraction_r2_at_least_050=(scores[use]>=.50).mean() if use.any() else np.nan))
    return rows
