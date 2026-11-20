"""Independently audit the NKI repeat-scan scores and the participant-level contrasts."""
from pathlib import Path
import json
from verify_results import SingularRidge, fixtures, join, sampled, stream
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nki-retest"
PENALTIES = 10.0 ** np.arange(-14, 5, dtype=float)
KEYS = ["half", "source_visit", "target_visit"]


def verify_close(actual, saved):
    """Allow absolute roundoff near zero and relative roundoff for unstable transfers."""
    np.testing.assert_allclose(actual, saved, rtol=2e-6, atol=5e-7)
    return float(np.max(np.abs(actual - saved) / np.maximum(1., np.abs(saved))))


def scores(predicted, target):
    mse = np.mean((predicted - target) ** 2, axis=0)
    return np.stack([1 - mse / target.var(axis=0), 3 * mse])


def check_aggregates(pairs, summary, participants):
    log = []
    for (half, visit, test_visit), group in pairs.groupby(KEYS):
        conditions = (half, visit, test_visit)
        info = summary.loc[conditions]
        target_report = participants.loc[conditions]
        differences = []
        for subject, target_data in group.groupby("target_subject"):
            errors = target_data.groupby("source_subject").normalized_mse.mean()
            own = errors.loc[subject]
            others = errors.drop(subject)
            r2 = target_data.groupby("source_subject").r2.mean()
            match = target_report.loc[subject]
            assert len(errors) == 25 and len(others) == 24
            rank = 1 + np.sum(others < own) + .5 * np.sum(others == own)
            np.testing.assert_allclose(
                [own, others.mean(), others.mean() - own, r2.loc[subject] - r2.drop(subject).mean(), rank],
                [match.self_nmse, match.other_mean_nmse, match.advantage_nmse, match.advantage_r2, match.decoder_rank],
                rtol=1e-12, atol=1e-8)
            differences.append(others.mean() - own)
        np.testing.assert_allclose(np.mean(differences), info.advantage_nmse, rtol=1e-12)
        assert (np.array(differences) > 0).sum() == info.n_better_than_mean_other
        log.append(f"{half}, visits {visit}->{test_visit}: mean other-minus-self normalized MSE "
                   f"{np.mean(differences):.9g}; top-ranked self decoders {int(info.n_decoder_top1)}/25.")
    return log


def check_primary(pairs, calibration, pooled, meta):
    with np.load(ROOT / "data" / "nki" / "nki_reference.npz", allow_pickle=False) as f:
        assert f["halves"][0] == "odd"
        ids = f["subjects"].astype(str).tolist()
        graphs = f["weights"][0].astype(float)
    assert meta["ordered_subjects"] == ids and len(ids) == 25
    count, n = len(ids), graphs.shape[-1]
    chosen = stream(10).choice(n, 8, replace=False)
    b = np.zeros(n)
    b[chosen[:4]] = .5
    b[chosen[4:]] = -.5
    assert {str(i + 1): b[i] for i in chosen} == meta["signed_inputs_one_based"]
    np.testing.assert_array_equal(np.where(b == 0)[0] + 1, meta["readouts_one_based"])
    measurements = calibration.query("half == 'odd'").set_index(["visit", "subject"]).sort_index()
    train, valid, models, tests = {}, {}, [], []
    visit_two_validation = []
    within_error = 0.
    for visit in range(2):
        for ix, person in enumerate(ids):
            a = graphs[visit, ix].copy()
            np.fill_diagonal(a, 0.)
            a *= .9 / np.max(np.abs(eigvalsh(a)))
            ident = 10_000 + visit * count + ix
            x, y = fixtures(a, b, ident, 0, 2352)
            validation = fixtures(a, b, ident, 1, 1176)
            model = SingularRidge((x, y))
            recorded = measurements.loc[(visit + 1, person)].sort_values("delay")
            alpha = model.tune(validation, PENALTIES)
            np.testing.assert_allclose(recorded.relative_ridge, alpha, rtol=1e-12, atol=0)
            target = join([fixtures(a, b, ident, 2, 512, block) for block in range(4)])
            got = scores(model.predict(target[0], alpha), target[1])
            within_error = max(within_error, verify_close(got, recorded[["r2", "normalized_mse"]].to_numpy().T))
            if visit == 0:
                train[person] = (x, y)
                valid[person] = validation
                models.append((model, alpha))
            else:
                tests.append(target)
                visit_two_validation.append(validation)

    ix_pairs = pairs.query("half == 'odd' and source_visit == 1").set_index(
        ["target_subject", "source_subject", "delay"]).sort_index()
    other_error = self_error = 0.
    for j, (x, y) in enumerate(tests):
        for i, (model, alpha) in enumerate(models):
            reported = ix_pairs.loc[(ids[j], ids[i])][["r2", "normalized_mse"]].to_numpy().T
            diff = verify_close(scores(model.predict(x, alpha), y), reported)
            if i == j:
                self_error = max(self_error, diff)
            else:
                other_error = max(other_error, diff)

    pool_scores = pooled.query("half == 'odd' and source_visit == 1").set_index("target_subject")
    pool_error = 0.
    for j, person in enumerate(ids):
        donor = meta["pooled_donor_rows"][person]
        assert person not in donor["final_fit"] and len(donor["final_fit"]) == count - 1
        assert set(donor["inner_fit"]).isdisjoint(donor["inner_validation"])
        assert set(donor["inner_fit"]) | set(donor["inner_validation"]) == set(donor["final_fit"])
        def collect(source, design):
            return join([sampled(source[pid], qty) for pid, qty in donor[design].items()])
        inner = SingularRidge(collect(train, "inner_fit"))
        alpha = inner.tune(collect(valid, "inner_validation"), PENALTIES)
        recorded = pool_scores.loc[person].sort_values("delay")
        np.testing.assert_allclose(recorded.pooled_alpha, alpha, rtol=1e-12, atol=0)
        model = SingularRidge(collect(train, "final_fit"))
        x, y = tests[j]
        pool_error = max(pool_error, verify_close(scores(model.predict(x, alpha), y),
                                                   recorded[["r2", "normalized_mse"]].to_numpy().T))

    log = [
        "Odd-direction diffusion subset. Independently applied the column-vector tanh recurrence and "
        "SciPy direct centered-design SVD. Reconstructed fit, validation and four disjoint test streams.",
        f"50/50 individual penalties reproduced; 500 within-scan delay scores checked, "
        f"maximum scaled discrepancy {within_error:.3g}.",
        f"Visit 1 to visit 2: all 25 same-person transfers and 600 other-person transfers, ten delays "
        f"each. Maximum scaled discrepancies: same person {self_error:.3g}, other person {other_error:.3g}.",
        f"25/25 leave-person-out shared penalties reproduced by design-matrix SVD; all 250 pooled "
        f"delay scores checked, maximum scaled discrepancy {pool_error:.3g}.",
        "Scaled discrepancy is |audited - reported| / max(1, |reported|), over both R2 and normalized MSE."
    ]
    log.extend(check_transfer_tuning(models, tests, visit_two_validation, ids))
    return log


def check_transfer_tuning(models, tests, validation, ids):
    base = OUT / "transfer-tuned"
    if not base.exists():
        return []
    def read(name):
        return pd.read_csv(base / (name + ".csv"),
                           dtype={"subject": str, "source_subject": str, "target_subject": str})
    paired = read("pair_scores")
    reported = paired.query("half == 'odd' and source_visit == 1").set_index(
        ["target_subject", "source_subject", "delay"]).sort_index()
    donors = read("donor_validation").query("half == 'odd' and source_visit == 1")
    donor_errors = []
    for j, (model, _) in enumerate(models):
        x, y = validation[j]
        errors = [scores(model.predict(x, a), y)[1].mean() for a in PENALTIES]
        verify_close(np.array(errors), donors[donors.subject == ids[j]].sort_values("relative_ridge").validation_nmse.to_numpy())
        donor_errors.append(errors)
    donor_errors = np.asarray(donor_errors)
    worst = 0.
    for j, (x, y) in enumerate(tests):
        for i, (model, _) in enumerate(models):
            eligible = [p for p in range(len(ids)) if p not in {i, j}]
            assert len(eligible) == (24 if i == j else 23)
            alpha = PENALTIES[np.argmin(np.mean(donor_errors[eligible], axis=0))]
            rows = reported.loc[(ids[j], ids[i])]
            np.testing.assert_allclose(rows.transfer_alpha, alpha, rtol=1e-12, atol=0)
            worst = max(worst, verify_close(scores(model.predict(x, alpha), y),
                                            rows[["r2","normalized_mse"]].to_numpy().T))
    summary = read("summary").set_index(KEYS).sort_index()
    people = read("participant_summary").set_index(KEYS + ["subject"]).sort_index()
    return [
        "Transfer-tuned sensitivity: on odd visit 1->2, independently recomputed all 475 cross-visit donor "
        "validation errors, excluded both members of each pair, reconstructed all 625 chosen penalties and "
        f"all 6,250 delay-specific transfers, maximum scaled discrepancy {worst:.3g}.",
        *["Transfer-tuned: " + row for row in check_aggregates(paired, summary, people)]
    ]


def main():
    labels = {"subject": str, "source_subject": str, "target_subject": str}
    def read(name):
        return pd.read_csv(OUT / (name + ".csv"), dtype=labels)
    pairs = read("pair_scores")
    expected = 4 * 25 * 25 * 10
    assert len(pairs) == expected and not pairs.duplicated(KEYS + ["source_subject", "target_subject", "delay"]).any()
    assert set(pairs.delay) == set(range(1, 11)) and pairs.groupby(KEYS).ngroups == 4
    summary = read("summary").set_index(KEYS).sort_index()
    people = read("participant_summary").set_index(KEYS + ["subject"]).sort_index()
    info = json.loads((OUT / "run_info.json").read_text(encoding="utf-8"))
    log = check_primary(pairs, read("within_scan_scores"), read("pooled_scores"), info)
    log.append("Independently recomputed 100 participant-level donor ranks and paired gaps over all 25,000 saved rows.")
    log.extend(check_aggregates(pairs, summary, people))
    report = "\n".join(log) + "\n"
    (OUT / "verification.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
