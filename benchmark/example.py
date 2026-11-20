"""Create a small synthetic repeat-connectome input for a smoke test."""
import argparse
from pathlib import Path
import numpy as np


def create(path):
    rng = np.random.default_rng(731)
    n, p = 5, 16
    core = rng.gamma(2, 1, size=(n,p,p))
    keep = rng.random(size=(n,p,p)) < .4
    upper = np.triu(core*keep, 1)
    repeats = np.triu(upper*rng.uniform(.8,1.2,size=upper.shape), 1)
    both = np.stack((upper,repeats))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, weights=both+both.swapaxes(-1,-2), sessions=["first","repeat"],
                            subjects=[f"synthetic_{i+1}" for i in range(n)], region_ids=np.arange(p))


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    create(cli.parse_args().output)
