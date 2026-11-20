"""Keep each NKI decoder, source graph and test input fixed across all four substitutions."""
import nki_numerical_repeat as numerical
import numpy as np
import pandas as pd


ROOT = numerical.ROOT
OUT = ROOT / "results/portability-path"
pilot = numerical.PILOT


def score(net, model, mapping, stream, labels):
    states, test_labels = pilot.combine([
        pilot.simulate(net, mapping, stream, numerical.ROLE, pilot.TEST_N, k)
        for k in range(pilot.TEST_STREAMS)
    ])
    np.testing.assert_array_equal(test_labels, labels)
    prediction = model.predict(states, numerical.PENALTY)
    return float((1 - ((labels-prediction)**2).mean(axis=0) / labels.var(axis=0)).mean())


def matched_sources():
    subjects, visits, halves, ref, reruns = numerical.fetch_data(ROOT/"data/nki/inputs.h5")
    odd, even = list(halves).index("odd"), list(halves).index("even")
    mapping = pilot.input_map(ref.shape[-1], key=10)
    repeat_nets = [numerical.reservoir(raw) for raw in ref[odd, 1]]
    upper = np.triu_indices(ref.shape[-1], k=1)
    rows = []
    for i, person in enumerate(subjects):
        source_raw = ref[odd, 0, i]
        source_net = numerical.reservoir(source_raw)
        train_stream = 10000+i
        test_stream = train_stream+len(subjects)
        training = pilot.simulate(source_net, mapping, train_stream, 0, pilot.FIT_N)
        model = pilot.RidgeFamily.from_design(training)
        states, labels = pilot.combine([
            pilot.simulate(source_net, mapping, test_stream, numerical.ROLE, pilot.TEST_N, k)
            for k in range(pilot.TEST_STREAMS)
        ])
        predictions = model.predict(states, numerical.PENALTY)
        source_score = float((1 - ((labels-predictions)**2).mean(axis=0) / labels.var(axis=0)).mean())
        own, strangers = None, []
        for j, net in enumerate(repeat_nets):
            result = score(net, model, mapping, test_stream, labels)
            if j == i:
                own = result
            else:
                strangers.append(result)
        same_visit = score(numerical.reservoir(ref[even, 0, i]), model, mapping, test_stream, labels)
        cache = {source_raw[upper].tobytes(): source_score}
        variants = []
        for k in range(20):
            raw = reruns.loc[(person, visits[0,i], "odd", f"sim-{k}")]
            fingerprint = raw[upper].tobytes()
            if fingerprint not in cache:
                cache[fingerprint] = score(numerical.reservoir(raw), model, mapping, test_stream, labels)
            variants.append(cache[fingerprint])
        for rung, values in [("numerical_perturbation", variants), ("split_half", [same_visit]),
                             ("between_visit", [own]), ("stranger", strangers)]:
            rows.append(dict(source_id=person, rung=rung, n_targets=len(values), alpha=numerical.PENALTY,
                             test_stream=test_stream, source_r2=source_score, target_r2=np.mean(values)))
    people = pd.DataFrame(rows)
    forward = pd.read_csv(OUT/"visit_people.csv.gz", dtype={"target_id": str})
    forward = forward.query("cohort=='nki' and task=='linear' and rung=='between_visit' and alpha==10")
    forward = forward.set_index("target_id").sort_index()
    own = people.loc[people.rung.eq("between_visit")].set_index("source_id").sort_index()
    np.testing.assert_array_equal(own.index, forward.index)
    np.testing.assert_allclose(own[["source_r2","target_r2"]], forward[["source_r2","target_r2"]], atol=1e-12, rtol=0)
    return people


def summarize(people):
    rows = []
    for rung, block in people.groupby("rung", sort=False):
        values = block.sort_values("source_id")[["source_r2","target_r2"]].to_numpy()
        source, target = values.mean(axis=0)
        draws = np.random.default_rng(20261120).integers(len(block), size=(10000,len(block)))
        boot = values[draws].mean(axis=1)
        lo, hi = np.quantile(boot[:, 1]/boot[:, 0], [.025,.975])
        rows.append(dict(rung=rung, n_people=len(block), source_r2=source, target_r2=target,
                         ratio_of_means=target/source, ratio_lo=lo, ratio_hi=hi))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    people = matched_sources()
    people.to_csv(OUT/"matched_retention_people.csv", index=False)
    summary = summarize(people)
    summary.to_csv(OUT/"matched_retention_summary.csv", index=False)
    print(summary.to_string(index=False))
