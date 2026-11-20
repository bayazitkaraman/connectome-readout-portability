"""Score frozen readouts across independent structural MRI visits in NKI–Rockland."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_feasibility as pilot
import numpy as np
import pandas as pd


DEFAULT_DATA = ROOT / "data" / "nki" / "nki_reference.npz"
OUT = ROOT / "results" / "nki-retest"
GAIN = .90
PENALTIES = np.logspace(-14, 4, 19)


@dataclass
class FittedScan:
    training: tuple
    validation: tuple
    testing: tuple
    model: pilot.RidgeFamily
    penalty: float
    weights: np.ndarray
    within_nmse: np.ndarray
    within_r2: np.ndarray
    stream_id: int


def errors(predictions, testing):
    _, y = testing
    mse = np.mean(np.square(predictions - y), axis=0)
    variance = np.var(y, axis=0)
    if np.any(variance <= 0) or not np.all(np.isfinite(mse)):
        raise ValueError("Invalid test-target variance or prediction error.")
    return mse / pilot.SIGNAL_VAR, 1 - mse / variance


def fit_scan(raw, mapping, person_index, visit_index, count):
    # Use a separate identity for each acquisition. Reuse its inputs across diffusion halves.
    stream_id = 10000 + visit_index * count + person_index
    net = pilot.reservoir(raw, GAIN)
    training = pilot.simulate(net, mapping, stream_id, 0, pilot.FIT_N)
    validation = pilot.simulate(net, mapping, stream_id, 1, pilot.VALID_N)
    testing = pilot.combine([
        pilot.simulate(net, mapping, stream_id, 2, pilot.TEST_N, s)
        for s in range(pilot.TEST_STREAMS)
    ])
    model = pilot.RidgeFamily.from_design(training)
    if not np.isfinite(model.scale) or model.scale <= 0:
        raise ValueError("The graph and input placement produced no measurable state variance.")
    alpha = model.tune(validation, PENALTIES)
    weights = model.weights(alpha)
    nmse, r2 = errors(model.predict(testing[0], alpha), testing)
    return FittedScan(training, validation, testing, model, alpha, weights, nmse, r2, stream_id)


def shared_pool(scans, target_id, subjects):
    # The same donor split is reused for a target for both directions and diffusion halves.
    donors = np.delete(np.arange(len(scans)), target_id)
    shuffled = pilot.rng(70, target_id).permutation(donors)
    n_inner = int(round(len(donors) * 2 / 3))
    inner, valid = shuffled[:n_inner], shuffled[n_inner:]
    if not len(valid) or not len(inner):
        raise ValueError("Insufficient donors for grouped validation of the shared model.")
    assert target_id not in donors and not set(inner) & set(valid)

    def allocations(indices, total):
        base, remainder = divmod(total, len(indices))
        return [(int(i), base + (position < remainder)) for position, i in enumerate(indices)]

    def fit_from_donors(allocation):
        return pilot.RidgeFamily(pilot.merge([
            pilot.Moment.from_data(pilot.sampled(scans[i].training, rows))
            for i, rows in allocation
        ]))

    inner_fit = allocations(inner, pilot.FIT_N)
    inner_valid = allocations(valid, pilot.VALID_N)
    all_fit = allocations(donors, pilot.FIT_N)
    validation = pilot.combine([pilot.sampled(scans[i].validation, rows)
                                for i, rows in inner_valid])
    alpha = fit_from_donors(inner_fit).tune(validation, PENALTIES)
    log = {group: {str(subjects[i]): rows for i, rows in allocation}
           for group, allocation in (("inner_fit", inner_fit), ("inner_validation", inner_valid),
                                     ("final_fit", all_fit))}
    return fit_from_donors(all_fit), alpha, log


def load(path):
    with np.load(path, allow_pickle=False) as f:
        raw = f["weights"].astype(np.float64, copy=True)
        halves, sessions, subjects = [f[name].astype(str) for name in ("halves", "sessions", "subjects")]
        visit_codes = (f["session_labels"].astype(str) if "session_labels" in f
                       else np.repeat(sessions[:, None], len(subjects), axis=1))
    if (len(sessions) != 2 or len(set(subjects)) != len(subjects)
            or list(halves) != ["odd", "even"] or len(set(sessions)) != 2):
        raise ValueError("Expected unique people and visits and ordered odd/even diffusion subsets.")
    if (raw.ndim != 5 or raw.shape[:3] != (len(halves), len(sessions), len(subjects))
            or raw.shape[3] != raw.shape[4] or raw.shape[3] <= 8):
        raise ValueError("Weights must be ordered by diffusion subset, visit, subject and two parcel axes.")
    if not np.array_equal(raw, raw.swapaxes(-1, -2)) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Expected finite nonnegative undirected connectomes.")
    if visit_codes.shape != (2, len(subjects)) or (visit_codes[0] == visit_codes[1]).any():
        raise ValueError("Every participant needs two distinct recorded visit identifiers.")
    idx = np.arange(raw.shape[-1])
    raw[..., idx, idx] = 0
    return raw, halves, sessions, subjects, visit_codes


def run(path, out):
    raw, halves, session_ids, subjects, visit_codes = load(path)
    count, nodes = len(subjects), raw.shape[-1]
    mapping = pilot.input_map(nodes, key=10)
    i, j = np.triu_indices(nodes, 1)
    paired, within, pooled = [], [], []
    donor_metadata = {}
    max_prediction_difference = 0.
    for half, half_graphs in zip(halves, raw):
        scans = [[fit_scan(g, mapping, p, v, count) for p, g in enumerate(visit)]
                 for v, visit in enumerate(half_graphs)]
        edge_vectors = half_graphs[:, :, i, j]
        correlations = np.corrcoef(edge_vectors.reshape(2 * count, len(i)))
        if not np.isfinite(correlations).all():
            raise ValueError("A connectome has undefined edge-weight correlation.")
        for v, group in enumerate(scans):
            for p, scan in enumerate(group):
                for k, delay in enumerate(pilot.DELAYS):
                    within.append(dict(half=half, visit=v+1, subject=subjects[p], delay=delay,
                                       visit_code=visit_codes[v,p],
                                       normalized_mse=scan.within_nmse[k], r2=scan.within_r2[k],
                                       relative_ridge=scan.penalty, stream_id=scan.stream_id))

        for source_visit, target_visit in [(0, 1), (1, 0)]:
            source_scans, target_scans = scans[source_visit], scans[target_visit]
            for tp, target in enumerate(target_scans):
                target_x = target.testing[0]
                pool_model, pool_alpha, log = shared_pool(source_scans, tp, subjects)
                donor_metadata[str(subjects[tp])] = log
                nmse, r2 = errors(pool_model.predict(target_x, pool_alpha), target.testing)
                for k, delay in enumerate(pilot.DELAYS):
                    pooled.append(dict(half=half, source_visit=source_visit+1, target_visit=target_visit+1,
                                       target_subject=subjects[tp], delay=delay, normalized_mse=nmse[k],
                                       target_visit_code=visit_codes[target_visit,tp],
                                       r2=r2[k], target_own_r2=target.within_r2[k],
                                       target_own_nmse=target.within_nmse[k], pooled_alpha=pool_alpha))
                for sp, source in enumerate(source_scans):
                    frozen = (target_x - source.model.mx) @ source.weights + source.model.my
                    if sp == 0 and tp == 0:
                        # Check the fast frozen-weight path against the original scoring method.
                        direct = source.model.predict(target_x, source.penalty)
                        diff = float(np.max(np.abs(frozen - direct)))
                        np.testing.assert_allclose(frozen, direct, atol=1e-10, rtol=1e-12)
                        max_prediction_difference = max(max_prediction_difference, diff)
                    nmse, r2 = errors(frozen, target.testing)
                    corr = correlations[source_visit * count + sp, target_visit * count + tp]
                    for k, delay in enumerate(pilot.DELAYS):
                        paired.append(dict(half=half, source_visit=source_visit+1, target_visit=target_visit+1,
                                           source_subject=subjects[sp], target_subject=subjects[tp], delay=delay,
                                           source_visit_code=visit_codes[source_visit,sp],
                                           target_visit_code=visit_codes[target_visit,tp],
                                           normalized_mse=nmse[k], r2=r2[k], wiring_correlation=corr,
                                           source_own_r2=source.within_r2[k], target_own_r2=target.within_r2[k],
                                           source_own_nmse=source.within_nmse[k], target_own_nmse=target.within_nmse[k],
                                           source_alpha=source.penalty, target_alpha=target.penalty))
        print(f"{half}: scored {count} x {count} decoder transfers per direction", flush=True)

    out.mkdir(exist_ok=True, parents=True)
    for name, rows, expected in [("pair_scores", paired, 4*count*count*len(pilot.DELAYS)),
                                 ("pooled_scores", pooled, 4*count*len(pilot.DELAYS)),
                                 ("within_scan_scores", within, 4*count*len(pilot.DELAYS))]:
        if len(rows) != expected:
            raise RuntimeError(f"{name} row count does not match study design.")
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    w, readouts = mapping
    metadata = dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    gain=GAIN, task="Recover deterministic-seeded independent uniform inputs at delays 1-10.",
                    fit_rows=pilot.FIT_N, tune_rows=pilot.VALID_N,
                    test_streams=pilot.TEST_STREAMS, test_rows_per_stream=pilot.TEST_N,
                    warmup=pilot.WARMUP, seed=pilot.SEED, relative_ridge_grid=PENALTIES.tolist(),
                    simulations="SeedSequence([seed,11,10000+(visit-1)*n+subject_index,role,repeat]). "
                                "The subject index is zero-based in ordered_subjects; role is 0 (fit), "
                                "1 (validation) or 2 (test). Same visit and subject use identical "
                                "inputs across diffusion halves. Every source model uses only its "
                                "own training and tuning streams. Test sequences start with zero state.",
                    ordered_subjects=subjects.tolist(), visit_labels=dict(zip(["1","2"], session_ids.tolist())),
                    visit_codes_by_subject={str(s): visit_codes[:, p].tolist() for p, s in enumerate(subjects)},
                    brain_regions=nodes, signed_inputs_one_based={str(n+1):w[0,n] for n in np.flatnonzero(w)},
                    readouts_one_based=(readouts+1).tolist(), pooled_donor_rows=donor_metadata,
                    max_fast_vs_direct_prediction_difference=max_prediction_difference)
    (out / "run_info.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.data, args.out)
