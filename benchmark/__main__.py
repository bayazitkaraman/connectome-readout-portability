import argparse
import json
from pathlib import Path
import tempfile

from . import analysis


def main():
    cli = argparse.ArgumentParser(description="Benchmark frozen connectome readouts on aligned repeat scans.")
    cli.add_argument("input", type=Path, help="NPZ containing weights, subjects and sessions")
    cli.add_argument("output", type=Path, help="new directory for results")
    cli.add_argument("--half", help="one labeled reconstruction for a five-dimensional weights array")
    cli.add_argument("--task", choices=("recall", "product"), default="recall")
    cli.add_argument("--lags", nargs="+", type=int, help="distinct positive input delays. Product multiplies these targets")
    cli.add_argument("--seeds", nargs="+", type=int, default=[20261119])
    cli.add_argument("--penalties", nargs="+", type=float, default=analysis.DEFAULT_GRID.tolist())
    cli.add_argument("--gain", type=float, default=.9, help="target spectral radius")
    cli.add_argument("--input-count", type=int, default=8, help="even number of driven parcels, fewer than the parcel count")
    cli.add_argument("--map-key", type=int, default=10, help="random-stream key for the fixed input-node map")
    cli.add_argument("--warmup", type=int, default=200)
    cli.add_argument("--train-steps", type=int, default=2352)
    cli.add_argument("--validation-steps", type=int, default=1176)
    cli.add_argument("--test-steps", type=int, default=512, help="steps per independent test stream")
    cli.add_argument("--test-streams", type=int, default=4)
    cli.add_argument("--bin-edges", nargs="+", type=float, default=[.45,.6,.75,.85,.92],
                     help="internal cut points between -1 and 1")
    cfg = cli.parse_args()
    if cfg.output.exists():
        cli.error("Choose a new output directory. Existing results are preserved.")
    cfg.lags = cfg.lags or (list(range(1,11)) if cfg.task == "recall" else [1,2,3])
    try:
        data = analysis.load_data(cfg.input, cfg.half)
        analysis.validate_config(cfg, data[0].shape[-1])
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=cfg.output.parent) as tmp:
            # Stage results so interrupted runs do not appear as completed directories.
            stage = Path(tmp) / "result"
            stage.mkdir()
            info = analysis.run(data, cfg, stage)
            (stage / "run.json").write_text(json.dumps(info, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            stage.rename(cfg.output)
    except (ValueError, OSError, KeyError) as error:
        cli.exit(2, f"Benchmark error: {error}\n")
    print(f"Saved {cfg.task} benchmark to {cfg.output}")


if __name__ == "__main__":
    main()
