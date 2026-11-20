"""Resolve the primary R² threshold more finely within its original crossing brackets."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from analyze_nki_calibration import OUT, analyze_condition, crossings


def main(workers):
    original = pd.read_csv(OUT/"scores.csv", dtype={"subject":str})
    original = original[(original.half=="odd") & (original.source_visit==1)]
    coarse = original.groupby(["method","budget"]).r2.mean().unstack("budget")
    budgets = coarse.columns.to_numpy()
    levels = crossings(coarse.loc[["personal","pooled"]].to_numpy(), budgets, .5)
    if not (np.isfinite(levels).all() and np.all((levels>0)&(levels<=128))):
        print("This cohort did not meet the protocol's refinement condition.")
        return
    evaluate = set()
    for level in levels.astype(int):
        previous = budgets[budgets<level][-1]
        evaluate.update(range(int(previous),int(level)+1,4))
    folder = OUT/"refinement"
    args = [sys.executable,str(Path(__file__).with_name("nki_fewshot.py")),
            "--workers",str(workers),"--condition","odd-forward","--out",str(folder),
            "--budgets",*[str(k) for k in sorted(evaluate)]]
    subprocess.run(args,check=True)
    grid = pd.read_csv(folder/"scores.csv",dtype={"subject":str})
    keys = ["subject","repeat","method","budget"]
    if set(grid.budget)!=evaluate or len(grid)!=len(evaluate)*25*6*3 or grid.duplicated(keys).any():
        raise ValueError("Incomplete refinement grid")
    overlap = grid.merge(original,on=keys,validate="one_to_one",suffixes=("_new","_old"))
    for metric in ["r2","nmse"]:
        np.testing.assert_allclose(overlap[metric+"_new"],overlap[metric+"_old"],rtol=1e-9,atol=1e-9)
    merged = pd.concat([original,grid[~grid.budget.isin(budgets)]],ignore_index=True)
    estimates = analyze_condition(merged,dict(half="odd",source_visit=1,target_visit=2))
    for name in ["curves","thresholds","threshold_comparisons","participant_thresholds"]:
        frame = pd.DataFrame(estimates[name])
        if "threshold" in frame:
            frame = frame[frame.threshold==.50]
        frame.to_csv(folder/(name+".csv"),index=False)
    print(f"Reproduced {len(overlap)} overlapping observations; refined grid {sorted(evaluate)}.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    main(parser.parse_args().workers)
