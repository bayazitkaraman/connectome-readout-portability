"""Rebuild both cohorts' anatomical and recall contrasts and their correlations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "anatomy-advantage"
RULES = ("equal_donor", "fixed_025")


def edges_for(cohort, run):
    if cohort == "nki":
        path = ROOT / "data/nki/nki_reference.npz"
    else:
        path = ROOT / "data/replication_candidates/corr/bnu1_dkt_retest.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == run["input_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        subjects = arrays["subjects"].astype(str).tolist()
        weights = arrays["weights"]
        if cohort == "nki":
            assert list(arrays["sessions"]) == ["BAS1", "other"]
            assert (arrays["standard_session_labels"][0] == "BAS1").all()
            assert not (arrays["standard_session_labels"][1] == "BAS1").any()
            weights = weights[list(arrays["halves"]).index("odd")]
        else:
            assert list(arrays["sessions"]) == ["1", "2"]
    assert subjects == run["subjects"]
    assert list(weights.shape) == [2, len(subjects), 83, 83]
    np.testing.assert_array_equal(weights, weights.swapaxes(-1,-2))
    upper = np.triu_indices(83, k=1)
    assert len(upper[0]) == 3403
    vectors = weights[:, :, upper[0], upper[1]]
    if cohort == "bnu1":
        for j, person in enumerate(subjects):
            for t in [0,1]:
                raw_path = path.parent / "raw" / f"sub-{person}_ses-{t+1}_connectome.csv"
                raw_rows = np.loadtxt(raw_path)
                assert np.array_equal(raw_rows, np.floor(raw_rows))
                i1, i2, w = raw_rows.astype(np.int64).T
                raw = np.zeros((83,83), dtype=int)
                raw[i1-1, i2-1] = w
                np.testing.assert_array_equal(raw[upper], vectors[t,j])
    return subjects, vectors


def reconstruct_pairs(cohort):
    folder = RESULTS / f"{cohort}-augmentation"
    run = json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
    subjects, vectors = edges_for(cohort, run)
    scores = pd.read_csv(folder/"scores.csv", dtype={"subject":str, "other_subject":str})
    assert set(scores.half) == {"odd" if cohort == "nki" else "all"}
    assert set(scores.source_visit) == {1} and set(scores.target_visit) == {2}
    scores = scores[scores.budget == 8].set_index(
        ["subject","other_subject","candidate_draw","mix_rule","repeat","method"])
    assert scores.index.is_unique
    matches = np.random.default_rng(
        np.random.SeedSequence([run["seed"], 172, len(subjects)])
    ).permutation(len(subjects)).tolist()
    result = []
    for row in run["records"]:
        own, other, draw = row["subject"], row["other_subject"], row["candidate_draw"]
        j, k = subjects.index(own), subjects.index(other)
        assert matches[(matches.index(j) + draw + 1) % len(subjects)] == k
        own_r = np.corrcoef(vectors[1,j], vectors[0,j])[0,1]
        other_r = np.corrcoef(vectors[1,j], vectors[0,k])[0,1]
        for rule in RULES:
            paired_scores = []
            for repeat in range(6):
                key = (own, other, draw, rule, repeat)
                paired_scores.append([scores.at[(*key, method), "r2"]
                                      for method in ("pool_plus_own","pool_plus_other")])
            own_score, other_score = np.mean(paired_scores, axis=0)
            result.append(dict(cohort=cohort, mix_rule=rule, subject=own, other_subject=other,
                               candidate_draw=draw, own_similarity=own_r, other_similarity=other_r,
                               anatomy_advantage=own_r-other_r, own_r2=own_score, other_r2=other_score,
                               accuracy_advantage=own_score-other_score))
    return pd.DataFrame(result)


def weighted_ranks(values, counts):
    # Each bootstrap replicate gives a different multiplicity to each observed value.
    # A value's midrank is the total count below it plus half the size of its tie group.
    keys, inverse = np.unique(values, return_inverse=True)
    groups = np.eye(len(keys))[inverse]
    group_counts = counts @ groups
    return (np.cumsum(group_counts,axis=1) - .5*group_counts)[:, inverse]


def verify_scores(actual, written, keys, fields):
    a = actual.set_index(keys).sort_index()
    b = written.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[fields], b[fields], atol=1e-12, rtol=0)


def verify(cohort, pairs, stats_file):
    fields = ["own_similarity","other_similarity","anatomy_advantage","own_r2","other_r2","accuracy_advantage"]
    people = sorted(pairs.subject.unique())
    n = len(people)
    draws = np.random.default_rng(20261119).integers(0, n, (10000,n))
    counts = np.zeros((len(draws),n), dtype=float)
    np.add.at(counts, (np.arange(len(draws))[:, None], draws), 1)
    entries = []
    for rule in RULES:
        selected = pairs[pairs.mix_rule == rule]
        ordered = selected.sort_values(["subject","candidate_draw"])
        anatomy = ordered.anatomy_advantage.to_numpy().reshape(n,3)
        score = ordered.accuracy_advantage.to_numpy().reshape(n,3)
        x, y = anatomy.mean(axis=1), score.mean(axis=1)
        est = np.corrcoef(rankdata(x), rankdata(y))[0,1]
        rx = weighted_ranks(x, counts) - n/2
        ry = weighted_ranks(y, counts) - n/2
        boot = (counts*rx*ry).sum(axis=1) / np.sqrt((counts*rx**2).sum(axis=1)*(counts*ry**2).sum(axis=1))
        entries.append((rule, "between_participants", est, boot))
        if rule == "fixed_025":
            xx, yy = anatomy - x[:,None], score - y[:,None]
            est = np.corrcoef(xx.ravel(),yy.ravel())[0,1]
            numerator = counts @ (xx*yy).sum(axis=1)
            denominator = np.sqrt((counts @ (xx**2).sum(axis=1))*(counts @ (yy**2).sum(axis=1)))
            entries.append((rule, "within_participant", est, numerator/denominator))
    report = []
    for rule, analysis, estimate, bootstrap in entries:
        valid = np.isfinite(bootstrap)
        lo, hi = np.quantile(bootstrap[valid], [.025,.975])
        reference = stats_file[(stats_file.cohort==cohort) & (stats_file.analysis==analysis) &
                               (stats_file.mix_rule==rule)]
        assert len(reference) == 1
        np.testing.assert_allclose([estimate,lo,hi], reference.iloc[0][["estimate","ci_low","ci_high"]].to_numpy(float),
                                   rtol=0, atol=2e-12)
        assert reference.iloc[0].valid_bootstrap_draws == valid.sum()
        report.append(dict(mix_rule=rule,analysis=analysis,estimate=estimate,ci=[lo,hi]))
    participants = pairs.groupby(["cohort","mix_rule","subject"], as_index=False)[fields].mean()
    return participants, dict(subjects=n, paired_comparisons=3*n, graph_edges=3403,
                              aggregate_score_rows_checked=3*n*6*2*2, estimates=report)


def main():
    stats_file = pd.read_csv(OUT/"statistics.csv")
    pair_file = pd.read_csv(OUT/"pairs.csv", dtype={"subject":str,"other_subject":str})
    people_file = pd.read_csv(OUT/"participants.csv", dtype={"subject":str})
    fields = ["own_similarity","other_similarity","anatomy_advantage","own_r2","other_r2","accuracy_advantage"]
    audits = {}
    for cohort in ("nki","bnu1"):
        pairs = reconstruct_pairs(cohort)
        participants, audits[cohort] = verify(cohort, pairs, stats_file)
        verify_scores(pairs, pair_file[pair_file.cohort==cohort],
                      ["mix_rule","subject","other_subject","candidate_draw"], fields)
        verify_scores(participants, people_file[people_file.cohort==cohort], ["mix_rule","subject"], fields)
    (OUT/"verification.json").write_text(json.dumps(audits, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
