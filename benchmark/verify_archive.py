"""Compare the generic CLI's forward, transfer-tuned predictions against archived study results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def compare(output, pairs, cohort, task):
    meta=json.loads((output/"run.json").read_text(encoding="utf-8"))
    if meta["settings"]["task"]!=task or 20261119 not in meta["settings"]["seeds"]:
        raise ValueError("This check expects the study seed and the stated task.")
    with np.load(output/"matrix_seed20261119_visit0_to_1.npz", allow_pickle=False) as f:
        scores = f["fixed_r2"]
        selected = f["transfer_tuned_penalties"]
        pos = {s:i for i,s in enumerate(f["subjects"])}
        alphas = {float(a):i for i,a in enumerate(f["penalties"])}
    compared=set()
    max_error=0.
    with pairs.open(newline="",encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["cohort"]!=cohort or (task=="recall" and row["source_input"]!="original_distinct_inputs"):
                continue
            t,s=pos[row["target_id"]],pos[row["other_id"]]
            if t==s or (t,s) in compared:
                raise AssertionError("Duplicate or within-person archived pair.")
            a=float(row["alpha" if task=="product" else "relative_penalty"])
            if not np.isclose(a,selected[s,t],atol=0,rtol=1.e-12):
                raise AssertionError(f"Penalty differs for {row['target_id']}, {row['other_id']}.")
            # CSV representations may differ by one ULP from the original log grid.
            k=next(index for value,index in alphas.items() if np.isclose(a,value,atol=0,rtol=1e-12))
            diffs=np.abs([scores[s,t,k]-float(row["other_r2"]),scores[t,t,k]-float(row["own_r2"])])
            if not np.isfinite(diffs).all():
                raise AssertionError(f"Nonfinite score for {row['target_id']}, {row['other_id']}.")
            max_error=max(max_error,diffs.max())
            compared.add((t,s))
    if len(compared)!=len(pos)*(len(pos)-1) or max_error>1e-8:
        raise AssertionError(f"Archive comparison failed, pairs={len(compared)}, max error={max_error}.")
    return dict(cohort=cohort,task=task,pairs=len(compared),max_absolute_r2_error=max_error)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    cli.add_argument("cohort",choices=("nki","bnu1"))
    cli.add_argument("--task",choices=("recall","product"),default="recall")
    args=cli.parse_args()
    study=Path(__file__).resolve().parents[1]
    file=("results/density-controls/frozen/single_donor_pairs.csv" if args.task=="recall"
          else "results/nonlinear-recall/pair_scores.csv")
    print(json.dumps(compare(args.output,study/file,args.cohort,args.task),indent=2))
