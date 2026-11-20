"""Verify actual path rows with an independent recurrence, SVD fit and direct predictions."""
import hashlib
import json
from pathlib import Path

from verify_results import SingularRidge, fixtures, join, stream
from verify_nki_calibration import adjacency
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/portability-path"
PENALTIES = [10.0 ** exponent for exponent in range(-14, 9)] + [float("inf")]
KEYS = ["cohort", "task", "rung", "source_id", "target_id", "source_visit", "source_half",
        "target_visit", "target_half", "run", "alpha"]


def graph_pairs():
    """Select primary and reconstruction pairs by fixed position, without using performance."""
    for cohort, path in [
        ("nki", "nki/nki_reference.npz"),
        ("bnu1", "replication_candidates/corr/bnu1_dkt_retest.npz"),
    ]:
        with np.load(ROOT / "data" / path, allow_pickle=False) as archive:
            ids = archive["subjects"].astype(str)
            weights = archive["weights"]
            if cohort == "nki":
                halves = archive["halves"].tolist()
                sessions = archive["session_labels"]
                primary = weights[halves.index("odd")]
                half = "odd"
            else:
                primary = weights
                half = "all"
        n = len(ids)
        for t in (0, n // 2):
            for s in (t, (t + 1) % n):
                rung = "between_visit" if s == t else "stranger"
                key = (cohort, rung, ids[s], ids[t], 1, half, 2, half, -1)
                yield key, primary[0, s], primary[1, t], 10000 + s, 10000 + n + t

        if cohort != "nki":
            continue
        path = ROOT / "data/nki/inputs.h5"
        h = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(2 ** 20), b""):
                h.update(chunk)
        if h.hexdigest() != "9e084ace6509de76e35adba79319ce5aa2b53f299e1d9898fb6e7fd84ec28e8f":
            raise ValueError("Unexpected original NKI HDF SHA-256")
        raw = pd.read_hdf(path, key="graphs")
        for v, i, half in [(0, 0, "odd"), (1, n // 2, "even")]:
            source = weights[halves.index(half), v, i]
            other_half = "even" if half == "odd" else "odd"
            stream_id = 10000 + v * n + i
            key = (cohort, "split_half", ids[i], ids[i], v + 1, half, v + 1, other_half, -1)
            yield key, source, weights[halves.index(other_half), v, i], stream_id, stream_id
            selected = raw.loc[raw.pipeline.eq("det") & raw.subject.eq(ids[i]) &
                               raw.session.eq(sessions[v, i]) & raw.directions.eq(half)]
            runs = selected.set_index("simulation").graph
            np.testing.assert_array_equal(runs["ref"], source)
            upper = np.triu_indices(len(source), k=1)
            seen = set()
            for run in range(20):
                graph = runs[f"sim-{run}"]
                changed = not np.array_equal(source[upper], graph[upper])
                if changed in seen:
                    continue
                seen.add(changed)
                key = (cohort, "numerical_perturbation", ids[i], ids[i], v + 1, half, v + 1, half, run)
                yield key, source, graph, stream_id, stream_id
            if len(seen) != 2:
                raise ValueError("Expected both an identical and a changed numerical reconstruction")


def endpoints(y):
    return np.column_stack([y, np.prod(y[:, :3], axis=1)])


def reference_scores():
    rows = []
    for key, original, destination, fit_id, test_id in graph_pairs():
        source, target = adjacency(original), adjacency(destination)
        inputs = np.zeros(len(source))
        sites = stream(10).choice(len(source), 8, replace=False)
        inputs[sites[:4]], inputs[sites[4:]] = .5, -.5
        xfit, yfit = fixtures(source, inputs, fit_id, 0, 2352)
        fit = SingularRidge((xfit, endpoints(yfit)))
        xs, ys = join([fixtures(source, inputs, test_id, 61, 512, k) for k in range(4)])
        xt, yt = join([fixtures(target, inputs, test_id, 61, 512, k) for k in range(4)])
        np.testing.assert_array_equal(ys, yt)
        y = endpoints(ys)
        for alpha in PENALTIES:
            def scores(x):
                pred = fit.ybar if np.isinf(alpha) else fit.predict(x, alpha)
                return 1 - ((y - pred) ** 2).mean(axis=0) / np.var(y, axis=0)
            left, right = scores(xs), scores(xt)
            for task, span in (("linear", slice(0, 10)), ("nonlinear", slice(10, 11))):
                rows.append((key[0], task, *key[1:], alpha, left[span].mean(), right[span].mean()))
    return pd.DataFrame(rows, columns=KEYS + ["source_reference", "target_reference"])


def main():
    outputs = []
    for stem in ("visit", "rung"):
        table = pd.read_csv(OUT / f"{stem}_pairs.csv.gz", float_precision="round_trip",
                            dtype={"source_id": str, "target_id": str})
        # np.logspace and scalar power can differ by one ULP for a grid label.
        finite = np.isfinite(table.alpha)
        exponents = np.rint(np.log10(table.loc[finite, "alpha"])).astype(int)
        np.testing.assert_allclose(table.loc[finite, "alpha"], 10. ** exponents, atol=0, rtol=1e-13)
        table.loc[finite, "alpha"] = exponents.map(lambda e: 10. ** int(e))
        if stem == "visit":
            table["run"] = -1
        outputs.append(table[KEYS + ["source_r2", "target_r2"]])
    archived = pd.concat(outputs, ignore_index=True)
    reference = reference_scores()
    paired = reference.merge(archived, on=KEYS, how="left", validate="one_to_one")
    if paired[["source_r2", "target_r2"]].isna().any().any():
        raise ValueError("Missing archived score for a reference comparison")
    details = []
    for (cohort, rung, task), group in paired.groupby(["cohort", "rung", "task"]):
        for endpoint in ("source", "target"):
            expected = group[f"{endpoint}_reference"].to_numpy()
            observed = group[f"{endpoint}_r2"].to_numpy()
            difference = np.abs(observed - expected)
            scaled = difference / np.maximum(1, np.abs(expected))
            worst = int(np.argmax(difference))
            details.append(dict(cohort=cohort, rung=rung, task=task, endpoint=endpoint,
                                scores=len(group), max_absolute=float(difference[worst]),
                                max_scaled=float(scaled.max()), worst_absolute_reference=float(expected[worst]),
                                max_absolute_for_r2_minus_one_to_one=float(
                                    difference[np.abs(expected) <= 1].max()),
                                worst_absolute_alpha=str(group.iloc[worst].alpha)))
            np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=1e-8)
    peaks = {k: max(row[k] for row in details) for k in [
        "max_absolute", "max_scaled", "max_absolute_for_r2_minus_one_to_one"]}
    report = dict(
        scope="Two fixed target indices per cohort with personal and adjacent stranger sources. "
              "Two NKI visit/half cells with the opposite half and first exact and first changed numerical reruns.",
        method="Raw archived matrices, independently scaled symmetric adjacency, explicit column-vector "
               "tanh recurrence, SciPy SVD ridge and explicit squared prediction errors on four fresh signals.",
        pairs=len(paired.drop_duplicates([k for k in KEYS if k not in ("task", "alpha")])),
        operating_points_per_task=len(PENALTIES), scores=2*len(paired),
        scaling="Absolute difference divided by max(1, absolute independent R²).",
        tolerance={"atol": 1e-8, "rtol": 1e-8}, **peaks, details=details)
    (OUT / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['scores']} source and target scores; maximum scaled discrepancy {peaks['max_scaled']:.3g}.")


if __name__ == "__main__":
    main()
