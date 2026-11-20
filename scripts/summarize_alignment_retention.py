"""Paired uncertainty for the share of source recall retained after alignment."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "portability-baseline"
SEED = 20261120


def main():
    people = pd.read_csv(TABLES / "people.csv", dtype={"target_id": str})
    records = []
    for cohort, group in people.groupby("cohort", sort=False):
        group = group.sort_values("target_id")
        scores = group[["source_r2", "untouched_r2", "aligned_r2"]].to_numpy()
        indices = np.random.default_rng(SEED).integers(len(group), size=(10000, len(group)))
        source, untouched, aligned = scores.mean(axis=0)
        draw_source, draw_untouched, draw_aligned = scores[indices].mean(axis=1).T
        if (draw_source <= 0).any():
            raise ValueError("A bootstrap draw has a nonpositive source reference.")
        gains_pp = 100.0 * (draw_aligned - draw_untouched) / draw_source
        lo, hi = np.percentile(gains_pp, [2.5, 97.5])
        records.append(dict(cohort=cohort, n=len(group),
                            untouched_pct=100.0 * untouched / source,
                            aligned_pct=100.0 * aligned / source,
                            gain_pp=100.0 * (aligned - untouched) / source,
                            gain_pp_low=lo, gain_pp_high=hi))
    summary = pd.DataFrame(records)
    summary.to_csv(TABLES / "retention_intervals.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
