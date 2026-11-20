"""Simulation and scoring for repeat-scan readout transfer."""
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from itertools import permutations
import platform

import run_feasibility as pilot
import numpy as np
from threadpoolctl import threadpool_limits

from .metrics import TIE_ATOL, bin_curve, identify, reliability, wiring_similarity

DEFAULT_GRID = np.logspace(-14, 4, 19)


def ids(values, n, field):
    if values.shape != (n,) or values.dtype.kind not in "UiuS":
        raise ValueError(f"{field} must contain one integer or text label per position.")
    result = values.astype(str)
    if any(not label.strip() for label in result) or len(set(result)) != n:
        raise ValueError(f"{field} must contain unique, nonempty labels.")
    return result


def load_data(path, half):
    with np.load(path, allow_pickle=False) as src:
        raw = src["weights"]
        if raw.ndim == 5:
            names = ids(src["halves"], raw.shape[0], "halves").tolist()
            if half not in names:
                raise ValueError(f"Choose --half from {names} for this reconstruction axis.")
            raw = raw[names.index(half)]
        elif half is not None:
            raise ValueError("--half applies only to a labeled five-dimensional weights array.")
        if raw.ndim != 4 or raw.shape[-1] != raw.shape[-2]:
            raise ValueError("Weights must have shape (visits, subjects, regions, regions).")
        v, n, p, _ = raw.shape
        if min(v, n) < 2 or p < 3:
            raise ValueError("Provide at least two distinct visits, two subjects and three regions.")
        if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Weights must be finite, nonnegative, real numbers.")
        subjects = ids(src["subjects"], n, "subjects")
        sessions = ids(src["sessions"], v, "sessions")
        regions = ids(src["region_ids"], p, "region_ids") if "region_ids" in src else np.arange(p).astype(str)
    weights = raw.astype(float)
    weights[..., np.arange(p), np.arange(p)] = 0
    magnitude = weights.max(axis=(-1,-2))
    if np.any(magnitude == 0):
        raise ValueError("Every scan must contain nonzero interregional connections.")
    # Avoid overflow/underflow in eigenvalue and edge-correlation calculations
    # for otherwise equivalent networks measured in very different units.
    extreme = (magnitude > 1.e100) | (magnitude < 1.e-100)
    weights[extreme] /= magnitude[extreme,None,None]
    if not np.allclose(weights, weights.swapaxes(-1,-2), rtol=1.e-10, atol=1.e-14):
        raise ValueError("Every structural matrix must be symmetric within floating-point tolerance.")
    weights = .5 * weights + .5 * weights.swapaxes(-1, -2)
    return weights, subjects, sessions, regions


def validate_config(cfg, n_regions):
    if not 2 <= cfg.input_count < n_regions or cfg.input_count % 2:
        raise ValueError("Input count must be even, at least two and smaller than the region count.")
    if not np.isfinite(cfg.gain) or cfg.gain <= 0:
        raise ValueError("Spectral radius must be finite and positive.")
    if min(cfg.lags) < 1 or len(set(cfg.lags)) != len(cfg.lags) or max(cfg.lags) > cfg.warmup:
        raise ValueError("Supply unique positive lags no larger than warmup.")
    if len(set(cfg.seeds)) != len(cfg.seeds) or min(cfg.seeds) < 0 or cfg.map_key < 0:
        raise ValueError("Seeds must be unique and nonnegative; map key must be nonnegative.")
    if len(set(cfg.penalties)) != len(cfg.penalties) or not np.isfinite(cfg.penalties).all() or min(cfg.penalties) <= 0:
        raise ValueError("Supply distinct, finite, positive relative penalties.")
    if min(cfg.train_steps, cfg.validation_steps, cfg.test_steps) < 2 or cfg.test_streams < 1:
        raise ValueError("Use at least two steps for each split and at least one test stream.")
    edges = np.asarray(cfg.bin_edges)
    if not np.all(np.diff(edges)>0) or not np.isfinite(edges).all() or edges[0]<=-1 or edges[-1]>=1:
        raise ValueError("Bin edges must increase strictly inside (-1, 1).")


def get_rng(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def make_map(seed, key, n, count):
    selected = get_rng(seed, key).choice(n, size=count, replace=False)
    w = np.zeros((1,n))
    w[0,selected[:count//2]] = .5
    w[0,selected[count//2:]] = -.5
    return w, np.flatnonzero(w[0]==0)


def simulate(net, mapping, cfg, seed, stream_id, role, length, repeat=0):
    signal = get_rng(seed, 11, stream_id, role, repeat).uniform(-1, 1, length+cfg.warmup)
    states = net.simulate(ext_input=signal[:,None], w_in=mapping[0], output_nodes=mapping[1])
    times = np.arange(cfg.warmup, len(signal))
    labels = signal[times[:,None] - np.asarray(cfg.lags)]
    if cfg.task == "product":
        labels = labels.prod(axis=1, keepdims=True)
    return np.ascontiguousarray(states[cfg.warmup:]), labels


@dataclass
class Evaluation:
    """Orthogonal compression of [states, intercept, labels] preserves residual norms."""
    design: np.ndarray
    labels: np.ndarray
    variance: np.ndarray
    n: int

    @classmethod
    def from_data(cls, data):
        x, y = data
        block = np.linalg.qr(np.column_stack((x, np.ones(len(y)), y)), mode="r")
        variance = y.var(axis=0)
        if np.any(variance<=0) or not np.isfinite(variance).all():
            raise ValueError("The held-out labels have undefined or zero variance.")
        return cls(block[:,:x.shape[1]+1], block[:,x.shape[1]+1:], variance, len(y))

    def mse(self, weights):
        # weights shape (penalty, slopes plus intercept, outputs).
        residual = self.design @ weights - self.labels
        return np.square(residual).sum(axis=1) / self.n

    def r2(self, weights):
        return 1 - self.mse(weights) / self.variance


@dataclass
class Scan:
    weights: np.ndarray
    local: int
    validation: Evaluation
    test: Evaluation


def observe_test(net, mapping, cfg, seed, stream_id):
    streams = [simulate(net, mapping, cfg, seed, stream_id, 61, cfg.test_steps, k)
               for k in range(cfg.test_streams)]
    return Evaluation.from_data(pilot.combine(streams))


def fit_scan(net, mapping, cfg, seed, stream_id):
    data = simulate(net, mapping, cfg, seed, stream_id, 0, cfg.train_steps)
    model = pilot.RidgeFamily.from_design(data)
    if not np.isfinite(model.scale) or model.scale <= 0:
        raise ValueError(f"No measurable readout variance for stream {stream_id}; check graph and input nodes.")
    val = Evaluation.from_data(simulate(net, mapping, cfg, seed, stream_id, 1, cfg.validation_steps))
    bank = []
    for a in cfg.penalties:
        w = model.weights(a)
        bank.append(np.vstack((w, model.my - model.mx @ w)))
    bank = np.stack(bank)
    local = int(np.argmin(val.mse(bank).mean(axis=1)))
    return Scan(bank, local, val, observe_test(net, mapping, cfg, seed, stream_id))


def transfer_indices(loss):
    """Tune only on paired validation streams of identities outside the comparison."""
    n = len(loss)
    choices = np.empty((n,n), dtype=int)
    positions = np.arange(n)
    for s in range(n):
        for t in range(s,n):
            keep = (positions != s) & (positions != t)
            choice = np.argmin(loss[keep].mean(axis=0))
            choices[s,t] = choices[t,s] = choice
    return choices


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def score_direction(source, target, source_nets, mapping, cfg, seed, target_visit):
    n = len(source)
    k = len(cfg.penalties)
    all_r2 = np.empty((n,n,k))
    matched_source = np.empty((n,k))
    validation_loss = np.empty((n,k))
    for s, fitted in enumerate(source):
        stream_id = 10000 + target_visit*n+s
        same_input_test = observe_test(source_nets[s], mapping, cfg, seed, stream_id)
        matched_source[s] = same_input_test.r2(fitted.weights).mean(axis=1)
        validation_loss[s] = target[s].validation.mse(fitted.weights).mean(axis=1)
        for t, destination in enumerate(target):
            all_r2[s,t] = destination.test.r2(fitted.weights).mean(axis=1)
    return all_r2, matched_source, validation_loss


def summarize_direction(subjects, wiring, all_r2, matched_source, validations, source, dest, cfg, context, tables):
    n = len(subjects)
    idx = np.arange(n)
    s_local = np.array([scan.local for scan in source])
    t_local = np.array([scan.local for scan in dest])
    t_r2 = np.array([scan.test.r2(scan.weights[[a]]).mean() for scan,a in zip(dest,t_local)])
    wiring_rank, wiring_top = identify(wiring)
    policies = [(f"fixed:{a:.17g}",np.full((n,n),i)) for i,a in enumerate(cfg.penalties)]
    policies.append(("source_local",np.broadcast_to(s_local[:,None], (n,n))))
    if n>=3:
        policies.append(("transfer_tuned",transfer_indices(validations)))
    matrices = {}
    for name, selections in policies:
        matrix = np.take_along_axis(all_r2, selections[:,:,None], axis=2)[:,:,0]
        if name in ("source_local","transfer_tuned"):
            matrices[name+"_r2"] = matrix
            matrices[name+"_penalties"] = np.asarray(cfg.penalties)[selections]
        rank, top = identify(matrix)
        prefix = {**context, "selection":name}
        for row in bin_curve(wiring, matrix, cfg.bin_edges):
            tables["similarity_curve"].append({**prefix, **row})
        people = []
        for t in idx:
            others = idx[idx!=t]
            alpha = selections[t,t]
            # The mean reference compares own and each stranger with the exact
            # same transfer penalty for that held-out pair when transfer-tuned.
            own_for_pairs = all_r2[t,t,selections[others,t]] if name=="transfer_tuned" else np.repeat(matrix[t,t],n-1)
            tied = others[wiring[others,t] >= wiring[others,t].max() - TIE_ATOL]
            own_source = matched_source[t,alpha]
            source_local = matched_source[t,s_local[t]]
            people.append(dict(**prefix, subject=subjects[t], chosen_own_penalty=cfg.penalties[alpha],
                own_wiring=wiring[t,t], own_wiring_rank=wiring_rank[t], own_wiring_top1=wiring_top[t],
                own_decoder_rank=rank[t], own_decoder_top1=top[t], own_r2=matrix[t,t],
                mean_other_r2=matrix[others,t].mean(), own_minus_other_paired=(own_for_pairs-matrix[others,t]).mean(),
                nearest_other_r2=matrix[tied,t].mean(), nearest_other_count=len(tied),
                source_local_r2=source_local, source_r2=own_source, target_refit_r2=t_r2[t],
                source_penalty_cost=source_local-own_source, frozen_wiring_drop=own_source-matrix[t,t]))
        tables["targets"].extend(people)
        means={key:np.mean([person[key] for person in people]) for key in
               ("own_wiring_top1","own_decoder_top1","own_r2","mean_other_r2","nearest_other_r2",
                "own_minus_other_paired","source_local_r2","source_r2","target_refit_r2","frozen_wiring_drop")}
        tables["summary"].append(dict(**prefix, n_people=n, **means))
    return matrices


def run(data, cfg, out):
    graphs, subjects, visits, regions = data
    n = len(subjects)
    tables = {key:[] for key in ("individual","reliability","targets","summary","similarity_curve")}
    maps = {}
    with threadpool_limits(limits=1):
        nets = [[pilot.reservoir(graph, cfg.gain) for graph in cohort] for cohort in graphs]
        for seed in cfg.seeds:
            mapping = make_map(seed, cfg.map_key, len(regions), cfg.input_count)
            maps[str(seed)] = {"weights":mapping[0][0].tolist(),"readout_indices":mapping[1].tolist()}
            scans = [[fit_scan(net,mapping,cfg,seed,10000+v*n+i) for i,net in enumerate(cohort)]
                     for v,cohort in enumerate(nets)]
            individual = []
            for v,cohort in enumerate(scans):
                scores = []
                for i,scan in enumerate(cohort):
                    r2 = scan.test.r2(scan.weights)
                    for ai,a in enumerate(cfg.penalties):
                        tables["individual"].append(dict(seed=seed, visit=visits[v], subject=subjects[i],
                            penalty=a, selected_locally=ai==scan.local, mean_r2=r2[ai].mean(),
                            **{f"r2_{j}":score for j,score in enumerate(r2[ai])}))
                    scores.append(np.r_[r2.mean(axis=1),r2[scan.local].mean()])
                individual.append(np.stack(scores))
            for sv,tv in permutations(range(len(visits)),2):
                info = dict(seed=seed, source_visit=visits[sv], target_visit=visits[tv])
                wiring = wiring_similarity(graphs[sv], graphs[tv])
                all_r2, same_input, val = score_direction(scans[sv],scans[tv],nets[sv],mapping,cfg,seed,tv)
                extra = summarize_direction(subjects,wiring,all_r2,same_input,val,scans[sv],scans[tv],cfg,info,tables)
                np.savez_compressed(out/f"matrix_seed{seed}_visit{sv}_to_{tv}.npz", subjects=subjects,
                                    visits=visits[[sv,tv]], penalties=cfg.penalties, wiring=wiring, fixed_r2=all_r2, **extra)
                if sv<tv:
                    names = [f"fixed:{a:.17g}" for a in cfg.penalties]+["source_local"]
                    for i,name in enumerate(names):
                        tables["reliability"].append(dict(**info, selection=name, n=n,
                            **reliability(individual[sv][:,i],individual[tv][:,i])))
    for name,rows in tables.items():
        write_rows(out/(name+".csv"), rows)
    return dict(settings={key:value for key,value in vars(cfg).items() if key not in ("input","output")},
                input_name=cfg.input.name, input_sha256=hashlib.sha256(cfg.input.read_bytes()).hexdigest(),
                python=platform.python_version(), libraries={k:version(k) for k in
                    ("numpy","scipy","scikit-learn","threadpoolctl")},
                conn2res_commit="3ccb7074261c910847dcd0164b00ff3b02bade90", subjects=subjects.tolist(),
                visits=visits.tolist(), regions=regions.tolist(), maps=maps,
                axis_order=["source subject","target subject","penalty (fixed_r2 only)"],
                output_labels=cfg.lags if cfg.task=="recall" else [cfg.lags],
                test_role=61, train_role=0, validation_role=1)
