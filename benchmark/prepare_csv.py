"""Prepare a repeat-scan NPZ from a scan manifest and labeled connectivity matrices."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from .analysis import validated_weights


def read_manifest(manifest):
    scans = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"subject", "session", "path"} <= set(fields):
            raise ValueError(f"{manifest}: expected unique column names including subject, session and path.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{manifest}, line {reader.line_num}: wrong number of fields.")
            subject, session, path = (row[key] for key in ("subject", "session", "path"))
            if not all(value.strip() for value in (subject, session, path)):
                raise ValueError(f"{manifest}, line {reader.line_num}: empty subject, session or path.")
            if (subject, session) in scans:
                raise ValueError(f"{manifest}, line {reader.line_num}: duplicate scan for {subject!r}, {session!r}.")
            scans[subject, session] = manifest.parent / path
    subjects = list(dict.fromkeys(subject for subject, _ in scans))
    sessions = list(dict.fromkeys(session for _, session in scans))
    if min(len(subjects), len(sessions)) < 2:
        raise ValueError("Provide at least two subjects and two sessions in the manifest.")
    for subject in subjects:
        for session in sessions:
            if (subject, session) not in scans:
                raise ValueError(f"{manifest}: missing scan for subject {subject!r}, session {session!r}.")
    return scans, subjects, sessions


def read_matrix(path, regions):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        columns = next(reader, [])[1:]
        if len(columns) < 3 or any(not s.strip() for s in columns) or len(set(columns)) != len(columns):
            raise ValueError("Expected at least three unique, nonempty parcel labels in the header.")
        if regions is None:
            regions = columns
        if set(columns) != set(regions):
            raise ValueError("Column parcel labels differ from those in the first scan.")
        row_indices = {label: i for i, label in enumerate(regions)}
        column_indices = [row_indices[label] for label in columns]
        seen = set()
        matrix = np.empty((len(regions), len(regions)))
        for row in reader:
            if len(row) != len(columns) + 1:
                raise ValueError(f"Line {reader.line_num}: expected a row label and {len(columns)} weights.")
            label = row[0]
            if label not in row_indices or label in seen:
                raise ValueError(f"Line {reader.line_num}: unexpected or duplicate row label {label!r}.")
            try:
                matrix[row_indices[label], column_indices] = row[1:]
            except ValueError as error:
                raise ValueError(f"Line {reader.line_num}: expected numeric weights for parcel {label!r}.") from error
            seen.add(label)
        if len(seen) != len(regions):
            raise ValueError(f"Missing row labels: {', '.join(sorted(set(regions) - seen))}.")
    # Keep the original CSV values on disk. The benchmark applies its own
    # diagonal handling and scaling when it opens the completed archive.
    validated_weights(matrix[None, :, :])
    return matrix, regions


def prepare(manifest, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output file; {output} already exists.")
    scans, subjects, sessions = read_manifest(manifest)
    regions = None
    for visit, session in enumerate(sessions):
        for person, subject in enumerate(subjects):
            path = scans[subject, session]
            try:
                matrix, regions = read_matrix(path, regions)
            except (csv.Error, ValueError, OSError) as error:
                raise ValueError(f"{path} (subject {subject!r}, session {session!r}): {error}") from error
            if visit == 0 and person == 0:
                weights = np.empty((len(sessions), len(subjects), *matrix.shape))
            weights[visit, person] = matrix
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "cohort.npz"
        np.savez_compressed(staged, weights=weights, subjects=subjects, sessions=sessions, region_ids=regions)
        # Publish a completed file without replacing another run's output.
        if os.name == "nt":
            staged.rename(output)
        else:
            os.link(staged, output)
    return weights.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with subject, session and matrix path columns")
    parser.add_argument("output", type=Path, help="new NPZ file to create")
    args = parser.parse_args()
    try:
        visits, people, regions, _ = prepare(args.manifest, args.output)
    except (csv.Error, OSError, ValueError) as error:
        parser.exit(2, f"CSV import error: {error}\n")
    print(f"Saved {args.output} with {visits} visits, {people} subjects and {regions} regions.")


if __name__ == "__main__":
    main()
