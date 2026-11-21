import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.prepare_csv import prepare


class PrepareCsvTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.graphs = self.root / "matrices"
        self.graphs.mkdir()
        self.manifest = self.root / "scans.csv"
        self.output = self.root / "study.npz"
        self.labels = ["03", "A", "b"]
        self.matrix = np.array([[7, 2, 4], [2, 8, 0], [4, 0, 9]])
        self.header = ["subject", "session", "path"]
        self.rows = []
        for i, (subject, session) in enumerate([("002", "second"), ("001", "first"),
                                              ("002", "first"), ("001", "second")]):
            name = f"scan_{i}.csv"
            file = self.graphs / name
            self.rows.append([subject, session, str(file) if i == 1 else f"matrices/{name}"])
            cols = [0, 1, 2] if i == 0 else [2, 0, 1]
            values = self.matrix + i
            self.write_csv(file, [["parcel", *(self.labels[j] for j in cols)],
                                  *[[self.labels[j], *values[j, cols]] for j in [1, 2, 0]]])
        self.write_csv(self.manifest, [self.header, *self.rows])

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    def test_both_parcel_axes_and_manifest_orders(self):
        self.assertEqual(prepare(self.manifest, self.output), (2, 2, 3, 3))
        with np.load(self.output, allow_pickle=False) as result:
            self.assertEqual(result["subjects"].tolist(), ["002", "001"])
            self.assertEqual(result["sessions"].tolist(), ["second", "first"])
            self.assertEqual(result["region_ids"].tolist(), self.labels)
            for visit, person, offset in [(0, 0, 0), (1, 1, 1), (1, 0, 2), (0, 1, 3)]:
                np.testing.assert_array_equal(result["weights"][visit, person], self.matrix + offset)
        previous = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), previous)

    def test_missing_and_duplicate_scan(self):
        for rows, message in [(self.rows[:-1], "missing scan"), (self.rows + self.rows[:1], "duplicate scan")]:
            with self.subTest(message=message):
                self.write_csv(self.manifest, [self.header, *rows])
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.manifest, self.output)
                self.assertFalse(self.output.exists())

    def test_reject_bad_parcel_labels_and_weights(self):
        header = ["", *self.labels]
        good = [[self.labels[i], *row] for i, row in enumerate(self.matrix)]
        cases = [
            ([["", "03", "A", "A"], *good], "unique"),
            ([["", "03", "a", "b"], *good], "Column parcel labels"),
            ([header, good[0], good[0], good[2]], "duplicate row"),
            ([header, *good[:-1]], "Missing row"),
            ([header, ["03", 7, 9, 4], *good[1:]], "symmetric"),
            ([header, ["03", 7, "NaN", 4], *good[1:]], "finite"),
            ([header, ["03", 7, -1, 4], *good[1:]], "nonnegative"),
            ([header, ["03", 0, 0, 0], ["A", 0, 0, 0], ["b", 0, 0, 0]], "nonzero interregional"),
        ]
        for rows, message in cases:
            with self.subTest(message=message):
                self.write_csv(self.graphs / "scan_1.csv", rows)
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.manifest, self.output)
                self.assertFalse(self.output.exists())
