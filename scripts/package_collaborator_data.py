"""Bundle the brief and derived tables, including scan densities for an independent check."""
import csv
from io import StringIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/anatomy-advantage"
INPUTS = {"nki": ROOT / "data/nki/nki_reference.npz",
          "bnu1": ROOT / "data/replication_candidates/corr/bnu1_dkt_retest.npz"}


def scan_densities(path, cohort):
    with np.load(path, allow_pickle=False) as source:
        ids = source["subjects"].astype(str)
        weights = source["weights"]
        if cohort == "nki":
            weights = weights[source["halves"].astype(str).tolist().index("odd")]
        a, b = np.triu_indices(weights.shape[-1], k=1)
        density = (weights[:, :, a, b] > 0).mean(axis=-1)
    assert density.shape == (2,len(ids)) and len(a) == 3403
    assert np.all((density > 0) & (density <= 1))
    return {person: density[:, i] for i, person in enumerate(ids)}


def matches_for_handoff():
    by_cohort = {cohort: scan_densities(path, cohort) for cohort, path in INPUTS.items()}
    with (RESULTS / "pairs.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        densities = by_cohort[row["cohort"]]
        own_first, second = densities[row["subject"]]
        other_first = densities[row["other_subject"]][0]
        row.update(own_first_density=own_first, other_first_density=other_first, target_second_density=second)
    assert len(rows) == 2 * 3 * sum(len(s) for s in by_cohort.values())
    file = StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return file.getvalue()


if __name__ == "__main__":
    out = ROOT / "output/project1-collaborator-handoff.zip"
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.write(ROOT / "output/pdf/project1-collaborator-brief.pdf", "project1-collaborator-brief.pdf")
        bundle.write(RESULTS / "README.md", "README.md")
        for name in ("participants.csv", "statistics.csv", "descriptives.csv", "verification.json"):
            bundle.write(RESULTS / name, name)
        bundle.write(ROOT / "ANATOMY_ADVANTAGE_PROTOCOL.md", "ANATOMY_ADVANTAGE_PROTOCOL.md")
        bundle.write(ROOT / "POOL_AUGMENTATION_PROTOCOL.md", "POOL_AUGMENTATION_PROTOCOL.md")
        bundle.write(ROOT / "NKI_CALIBRATION_PROTOCOL.md", "NKI_CALIBRATION_PROTOCOL.md")
        bundle.writestr("matched_pairs.csv", matches_for_handoff())
        for folder in ("density-controls", "nki-split-half", "transfer-similarity"):
            base = ROOT / "results" / folder
            for file in sorted(base.rglob("*")):
                if file.is_file() and file.suffix in {".csv", ".md", ".json"}:
                    bundle.write(file, folder + "/" + file.relative_to(base).as_posix())
        bundle.write(ROOT / "DENSITY_CHECK_PROTOCOL.md", "DENSITY_CHECK_PROTOCOL.md")
        bundle.write(ROOT / "NKI_SPLIT_HALF_PROTOCOL.md", "NKI_SPLIT_HALF_PROTOCOL.md")
    with ZipFile(out) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) >= {
            "matched_pairs.csv", "participants.csv", "project1-collaborator-brief.pdf",
            "nki-split-half/pair_scores.csv", "nki-split-half/verification.json",
            "nki-split-half/single_scan_32_summary.json", "NKI_SPLIT_HALF_PROTOCOL.md",
            "transfer-similarity/README.md", "transfer-similarity/source_self_summary.csv",
            "transfer-similarity/nearest_donor_summary.json", "transfer-similarity/grid_summary.csv",
        }
        pairs = list(csv.DictReader(StringIO(bundle.read("matched_pairs.csv").decode())))
        assert len(pairs) == 492
    print(out)
