"""Check the forward native CLI state comparison against the independent study pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def verify():
    shifts = pd.read_csv(RESULTS / "portability-path/shift_people.csv", dtype={"source_id": str})
    align = pd.read_csv(RESULTS / "portability-baseline/people.csv", dtype={"target_id": str})
    report = {}
    for cohort, directory, penalty in (("nki", "cli-state-nki", 10.), ("bnu1", "cli-state-beijing", 1.)):
        run = json.loads((RESULTS / directory / "run.json").read_text())
        visits = list(map(str, run["visits"]))
        cli = pd.read_csv(RESULTS / directory / "state_pairs.csv",
                          dtype={"subject": str, "source_visit": str, "target_visit": str})
        forward = cli[(cli.source_visit == visits[0]) & (cli.target_visit == visits[1])].set_index("subject")
        ref = shifts[(shifts.cohort == cohort) & (shifts.rung == "between_visit")
                     & (shifts.task == "linear") & (shifts.alpha == penalty)].set_index("source_id")
        aligned = align[align.cohort == cohort].set_index("target_id")
        people = run["subjects"]
        assert sorted(people) == sorted(forward.index) == sorted(ref.index) == sorted(aligned.index)
        assert forward.penalty.eq(penalty).all()
        errors = {}
        for field, reference in (("displacement_pct",ref.shift_pct), ("source_r2",ref.source_r2),
                                 ("repeat_r2",ref.target_r2), ("isotropic_r2",ref.isotropic_r2),
                                 ("spatial_loss",ref.actual_output_shift), ("error_cross_term",ref.error_cross_term),
                                 ("aligned_r2",aligned.aligned_r2)):
            difference = forward.loc[people, field].to_numpy() - reference.loc[people].to_numpy()
            np.testing.assert_allclose(difference,0,rtol=0,atol=1e-10)
            errors[field] = float(np.abs(difference).max())
        report[cohort] = dict(people=len(people), input_sha256=run["input_sha256"], max_absolute_error=errors)
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
