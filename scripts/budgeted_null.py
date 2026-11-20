"""Reassign cohort edge profiles within fixed strata under two training-set budgets."""
import numpy as np

from profile_null import _validate_graphs


def budgeted_null(graphs, training_ids, keys, seed):
    """Swap disjoint pairs under a 15% mean regional strength and degree budget.

    Each statistic is the mean over training people of the L1 change in their
    regional values divided by their original total. Every person receives
    exactly the same edge permutation, including people outside training_ids.
    Inputs are undirected, nonnegative graphs with zero diagonal. Keys follow
    NumPy's strict upper triangle. Empty graphs contribute zero to both shifts.
    """
    graphs = np.asarray(graphs)
    ids = np.asarray(training_ids)
    if (graphs.ndim != 3 or ids.ndim != 1 or not ids.size
            or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= len(graphs) or np.unique(ids).size != ids.size):
        raise ValueError("Supply unique, in-range indices for the inner training people.")
    train = _validate_graphs(graphs[ids])
    if (train < 0).any():
        raise ValueError("Regional strength budgets require nonnegative edge weights.")
    n = train.shape[1]
    i, j = np.triu_indices(n, 1)
    keys = np.asarray(keys)
    if keys.shape != i.shape or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError("Provide an integer stratum key for each upper-triangle edge.")

    # Dimensions: edge, statistic (strength then degree), training person.
    weights = train[:, i, j].T.astype(np.float64)
    profiles = np.stack((weights, weights > 0), axis=1)
    totals = 2 * profiles.sum(axis=0)
    profiles /= np.where(totals > 0, totals, 1)
    source = np.arange(len(i))
    rng = np.random.default_rng(seed)
    proposals = []
    for key in np.unique(keys):
        slots = rng.permutation(np.flatnonzero(keys == key))
        proposals.extend(zip(slots[::2], slots[1::2]))
    rng.shuffle(proposals)

    regional_change = np.zeros((n, 2, len(ids)))
    current_shift = np.zeros(2)
    for left, right in proposals:
        a = {i[left], j[left]}
        b = {i[right], j[right]}
        # Swapping incident edges leaves their common endpoint unchanged.
        gains, losses = sorted(a - b), sorted(b - a)
        affected = gains + losses
        signs = np.array([1] * len(gains) + [-1] * len(losses))[:, None, None]
        old = regional_change[affected]
        proposed = old + signs * (profiles[right] - profiles[left])
        shift = current_shift + (np.abs(proposed) - np.abs(old)).sum(axis=0).mean(axis=1)
        if np.all(shift <= .15):
            regional_change[affected] = proposed
            current_shift = shift
            source[left], source[right] = right, left

    rewired = np.zeros_like(graphs)
    rewired[:, i, j] = graphs[:, i[source], j[source]]
    rewired[:, j, i] = rewired[:, i, j]
    return rewired
