"""Population nulls that move full cross-person edge-weight profiles together.

Upper-triangle indices always follow np.triu_indices(n_regions, 1). A shared
edge permutation exactly preserves each person's multiset of edge weights,
and all cross-person inner products, Euclidean distances and Pearson
correlations on the unnormalized upper-triangle edge vectors, up to roundoff.
Regional degrees, strengths, spectra and dynamical task performance can change.
"""
import numpy as np


def _validate_graphs(graphs):
    graphs = np.asarray(graphs)
    if (graphs.ndim != 3 or graphs.shape[0] == 0 or
            graphs.shape[1] != graphs.shape[2] or graphs.shape[1] < 2):
        raise ValueError("Expected at least one square graph with at least two nodes.")
    if not np.isfinite(graphs).all():
        raise ValueError("Edge weights must be finite.")
    if not np.array_equal(graphs, graphs.transpose(0, 2, 1)):
        raise ValueError("Graphs must be symmetric.")
    if np.any(graphs.diagonal(axis1=1, axis2=2)):
        raise ValueError("Zero the diagonal before constructing these nulls.")
    return graphs


def _bins(values, count, separate_zeros=False):
    """Quantile bins with no arbitrary splitting of tied numerical values."""
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("A requested bin count must be a positive integer.")
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite binning values.")
    result = np.zeros(values.shape, dtype=int)
    use = (values != 0) if separate_zeros else np.ones(values.shape, dtype=bool)
    if use.any():
        sample = values[use]
        cuts = np.unique(np.quantile(sample, np.arange(1, count) / count))
        result[use] = int(separate_zeros) + np.searchsorted(cuts, sample, side="right")
    return result


def build_strata(training_graphs, *, coordinates=None, hemispheres=None,
                 distance_bins=4, mean_weight_bins=0, support_bins=0,
                 endpoint_strength_bins=0):
    """Construct edge strata using only the provided training people and atlas.

    Coordinates and hemisphere labels are optional and must match matrix order.
    Distances are binned by quantile across atlas edges; hemisphere strata are
    unordered endpoint-label pairs, distinguishing (e.g.) LL, LR and RR. If
    requested, bin training edge mean weight, prevalence of nonzero weights,
    or the unordered pair of training-average node-strength bins. Zero-valued
    training edge means or prevalences receive their own category in addition
    to the requested number of nonzero bins. A zero bin count disables the
    corresponding feature. Without features there is a single stratum.
    """
    g = _validate_graphs(training_graphs)
    n = g.shape[1]
    i, j = np.triu_indices(n, 1)
    features = []
    if coordinates is not None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[0] != n or not np.isfinite(coords).all():
            raise ValueError("Coordinates must be finite and have one row per node.")
        if distance_bins:
            features.append(_bins(np.linalg.norm(coords[i] - coords[j], axis=1),
                                  distance_bins))
    if hemispheres is not None:
        h = np.asarray(hemispheres)
        if h.shape != (n,):
            raise ValueError("Provide one hemisphere label per node.")
        h = np.unique(h, return_inverse=True)[1]
        features.extend([np.minimum(h[i], h[j]), np.maximum(h[i], h[j])])
    if mean_weight_bins or support_bins:
        edge_values = g[:, i, j]
        if mean_weight_bins:
            features.append(_bins(edge_values.mean(axis=0), mean_weight_bins, True))
        if support_bins:
            features.append(_bins((edge_values != 0).mean(axis=0), support_bins, True))
    if endpoint_strength_bins:
        strength = g.sum(axis=2).mean(axis=0)
        b = _bins(strength, endpoint_strength_bins)
        features.extend([np.minimum(b[i], b[j]), np.maximum(b[i], b[j])])
    if not features:
        return np.zeros(len(i), dtype=int)
    return np.unique(np.column_stack(features), axis=0, return_inverse=True)[1].ravel()


def make_profile_null(graphs, keys, seed, return_permutation=False, *, partial_pairs=False):
    """Apply one shared profile permutation, independently within each stratum.

    Supply the same keys and seed to transform training and held-out graphs.
    Entries may legitimately remain in place, including all singleton strata.
    With partial_pairs, propose disjoint pairs and exchange each with probability 0.5.
    If requested, permutation[destination] is the source edge index.
    """
    g = _validate_graphs(graphs)
    i, j = np.triu_indices(g.shape[1], 1)
    keys = np.asarray(keys)
    if keys.shape != i.shape or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError("Provide an integer stratum key for each upper-triangle edge.")
    random = np.random.default_rng(seed)
    source = np.arange(len(i))
    for key in np.unique(keys):
        within = np.flatnonzero(keys == key)
        shuffled = random.permutation(within)
        if partial_pairs:
            pairs = shuffled[:2 * (len(within) // 2)].reshape(-1, 2)
            left, right = pairs[random.random(len(pairs)) < .5].T
            source[left], source[right] = right, left
        else:
            source[within] = shuffled
    null = np.zeros_like(g)
    null[:, i, j] = g[:, i[source], j[source]]
    null[:, j, i] = null[:, i, j]
    return (null, source) if return_permutation else null
