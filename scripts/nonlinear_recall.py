"""Evaluate nonlinear delayed-product readout portability under the locked protocol."""
import json
from dataclasses import dataclass
from functools import lru_cache

import nki_transfer as transfer
from paired_pool_augmentation import read_graphs
import numpy as np
import pandas as pd


pilot = transfer.pilot
OUT = transfer.ROOT / "results" / "nonlinear-recall"
BOOTSTRAP_SEED = 20261120
TARGET_VARIANCE = (1 / 3) ** 3


def product_data(net, mapping, ident, role, length, repeat=0):
    x, delayed = pilot.simulate(net, mapping, ident, role, length, repeat)
    y = delayed[:, :3].prod(axis=1, keepdims=True)
    return x, y


def testing_data(net, mapping, ident):
    return pilot.combine([product_data(net, mapping, ident, 61, pilot.TEST_N, k)
                          for k in range(pilot.TEST_STREAMS)])


def score(model, alpha, test):
    x, y = test
    mse = np.mean(np.square(y - model.predict(x, alpha)))
    return float(1 - mse / np.var(y))


@dataclass
class Source:
    training: tuple
    validation: tuple
    model: object


def anatomical_similarity(graphs):
    upper = np.triu_indices(graphs.shape[-1], 1)
    edge = graphs[:, :, upper[0], upper[1]].astype(float)
    edge -= edge.mean(axis=-1, keepdims=True)
    edge /= np.linalg.norm(edge, axis=-1, keepdims=True)
    similarity = edge[1] @ edge[0].T
    if not np.isfinite(similarity).all():
        raise ValueError("Undefined anatomical similarity")
    return similarity


def sanity_check_labels(net, mapping):
    ident, role = 10000, 0
    _, observed = product_data(net, mapping, ident, role, 100)
    u = pilot.rng(11, ident, role, 0).uniform(-1, 1, 100 + pilot.WARMUP)
    t = np.arange(pilot.WARMUP, len(u))
    direct = u[t - 1] * u[t - 2] * u[t - 3]
    np.testing.assert_array_equal(direct, observed[:, 0])


def run_cohort(cohort):
    graphs, ids, half = read_graphs(cohort)
    n = len(ids)
    mapping = pilot.input_map(graphs.shape[-1], key=10)
    similarity = anatomical_similarity(graphs)
    sources, target_tests, source_tests = [], [], []
    local_scores, losses, validation_rows = [], [], []
    for j, subject in enumerate(ids):
        source_id, target_id = 10000 + j, 10000 + n + j
        source_net = pilot.reservoir(graphs[0, j], transfer.GAIN)
        target_net = pilot.reservoir(graphs[1, j], transfer.GAIN)
        if j == 0:
            sanity_check_labels(source_net, mapping)
        source_train = product_data(source_net, mapping, source_id, 0, pilot.FIT_N)
        source_valid = product_data(source_net, mapping, source_id, 1, pilot.VALID_N)
        target_train = product_data(target_net, mapping, target_id, 0, pilot.FIT_N)
        target_valid = product_data(target_net, mapping, target_id, 1, pilot.VALID_N)
        source_test = testing_data(source_net, mapping, target_id)
        target_test = testing_data(target_net, mapping, target_id)
        np.testing.assert_array_equal(source_test[1], target_test[1])
        source_model = pilot.RidgeFamily.from_design(source_train)
        target_model = pilot.RidgeFamily.from_design(target_train)
        target_alpha = target_model.tune(target_valid, transfer.PENALTIES)
        source_alpha = source_model.tune(source_valid, transfer.PENALTIES)
        y = target_test[1]
        variance = y.var()
        local_scores.append(dict(
            local_r2=score(target_model, target_alpha, target_test),
            local_alpha=target_alpha, source_local_alpha=source_alpha,
            source_local_r2=score(source_model, source_alpha, source_test),
            test_product_mean=y.mean(), test_product_var=variance,
            constant_zero_r2=1 - np.mean(y ** 2) / variance,
            target_mean_baseline_r2=0.0))
        losses_i = []
        for alpha in transfer.PENALTIES:
            mse = np.mean((source_model.predict(target_valid[0], alpha) - target_valid[1]) ** 2)
            losses_i.append(mse / TARGET_VARIANCE)
            validation_rows.append(dict(cohort=cohort, subject=subject,
                                        relative_penalty=alpha, validation_nmse=mse / TARGET_VARIANCE))
        losses.append(losses_i)
        sources.append(Source(source_train, source_valid, source_model))
        target_tests.append(target_test)
        source_tests.append(source_test)

    losses = np.asarray(losses)
    labels = np.arange(n)
    pair_rows, person_rows, pools = [], [], {}
    for t, subject in enumerate(ids):
        target_test = target_tests[t]
        pooled, pooled_alpha, allocations = transfer.shared_pool(sources, t, ids)
        assert sum(allocations["final_fit"].values()) == pilot.FIT_N
        assert str(subject) not in allocations["final_fit"]
        assert not set(allocations["inner_fit"]) & set(allocations["inner_validation"])
        pools[str(subject)] = allocations
        local = local_scores[t]

        @lru_cache(None)
        def personal_scores(alpha):
            return (score(sources[t].model, alpha, source_tests[t]),
                    score(sources[t].model, alpha, target_test))

        rows = []
        for d, donor in enumerate(ids):
            if d == t:
                continue
            tune = (labels != t) & (labels != d)
            assert tune.sum() == n - 2
            alpha = transfer.PENALTIES[np.argmin(losses[tune].mean(axis=0))]
            self_r2, own_r2 = personal_scores(alpha)
            foreign = score(sources[d].model, alpha, target_test)
            rows.append(dict(cohort=cohort, half=half, target_id=subject, other_id=donor,
                             own_train_stream=10000 + t, donor_train_stream=10000 + d,
                             test_stream=10000 + n + t, alpha=alpha, own_r2=own_r2,
                             other_r2=foreign, own_minus_other=own_r2 - foreign, source_self_r2=self_r2,
                             own_similarity=similarity[t, t], other_similarity=similarity[t, d]))
        block = pd.DataFrame(rows)
        nearest = block.other_similarity.eq(block.other_similarity.max())
        block["nearest_foreign"] = nearest
        pair_rows.append(block)
        closest = block.loc[nearest, "other_r2"].mean()
        mean_other = block.other_r2.mean()
        own = block.own_r2.mean()
        self_r2 = block.source_self_r2.mean()
        pool_r2 = score(pooled, pooled_alpha, target_test)
        person_rows.append(dict(cohort=cohort, half=half, target_id=subject, **local,
                                own_r2=own, mean_other_r2=mean_other, nearest_r2=closest,
                                pooled_r2=pool_r2, pooled_alpha=pooled_alpha,
                                source_self_r2=self_r2, source_r2_minus_own=self_r2 - own,
                                own_minus_mean_other=own - mean_other,
                                nearest_minus_mean_other=closest - mean_other,
                                nearest_minus_pooled=closest - pool_r2, own_minus_pooled=own - pool_r2,
                                local_minus_own=local["local_r2"] - own,
                                own_minus_nearest_matched=block.loc[nearest, "own_minus_other"].mean(),
                                nearest_ids="|".join(block.loc[nearest, "other_id"].astype(str)),
                                nearest_ties=int(nearest.sum()), foreign_donors=n - 1,
                                test_stream=10000 + n + t))
        personal_scores.cache_clear()
    pairs = pd.concat(pair_rows, ignore_index=True)
    assert len(pairs) == n * (n - 1) and not pairs.duplicated(["target_id", "other_id"]).any()
    numeric = pairs.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()
    return pairs, pd.DataFrame(person_rows), pd.DataFrame(validation_rows), pools


METRICS = ["local_r2", "source_local_r2", "source_self_r2", "own_r2", "mean_other_r2", "nearest_r2",
           "pooled_r2", "nearest_minus_mean_other", "nearest_minus_pooled", "own_minus_mean_other",
           "own_minus_pooled", "own_minus_nearest_matched", "source_r2_minus_own", "local_minus_own",
           "constant_zero_r2", "target_mean_baseline_r2"]


def summarize(people):
    rows = []
    for cohort, group in people.groupby("cohort", sort=False):
        n = len(group)
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(10000, n))
        for metric in METRICS:
            values = group[metric].to_numpy()
            lower, upper = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            rows.append(dict(cohort=cohort, metric=metric, n=n, mean=values.mean(),
                             ci_low=lower, ci_high=upper, median=np.median(values),
                             participants_positive=int((values > 0).sum())))
    return pd.DataFrame(rows)


def main():
    np.testing.assert_array_equal(pilot.DELAYS[:3], [1, 2, 3])
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, people, validations, pools = [], [], [], {}
    for cohort in ("nki", "bnu1"):
        print(f"Preparing nonlinear recall for {cohort}", flush=True)
        p, s, v, pools[cohort] = run_cohort(cohort)
        pairs.append(p)
        people.append(s)
        validations.append(v)
    participants = pd.concat(people, ignore_index=True)
    summary = summarize(participants)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(OUT / "pair_scores.csv", index=False)
    participants.to_csv(OUT / "participant_scores.csv", index=False)
    pd.concat(validations, ignore_index=True).to_csv(OUT / "validation_penalties.csv", index=False)
    (OUT / "pool_allocations.json").write_text(json.dumps(pools, indent=2) + "\n", encoding="utf-8")
    metadata = dict(protocol="NONLINEAR_RECALL_PROTOCOL.md", task="u[t-1]*u[t-2]*u[t-3]",
                    variance_theoretical=TARGET_VARIANCE, score="1 - test MSE / empirical test variance",
                    simulation_seed=pilot.SEED, seed_keys="pilot.rng(11, stream_id, role, repeat)",
                    mapping_key=10, radius=transfer.GAIN, warmup=pilot.WARMUP, training_role=0,
                    validation_role=1, held_out_role=61, held_out_repeats=list(range(pilot.TEST_STREAMS)),
                    train_rows=pilot.FIT_N, validation_rows=pilot.VALID_N, per_test_stream_rows=pilot.TEST_N,
                    target_labels_for_transfer=0, target_labels_for_local_reference=pilot.FIT_N,
                    target_validation_labels_for_local_reference=pilot.VALID_N,
                    ridge_grid=transfer.PENALTIES.tolist(),
                    single_selection="Mean normalized MSE on paired visits of every remaining identity",
                    pooled_selection="Grouped source-visit donor validation with original shared_pool split",
                    within_target="Single donor cell per foreign donor, including anatomically closest in mean",
                    nearest_ties="Average the tied donors; exact comparison of anatomical correlations",
                    bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=10000,
                    bootstrap="Percentile of paired resampled target people; donor bank and test signals fixed")
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
