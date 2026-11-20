# Readout transfer across repeat structural connectomes

How much of a decoder’s accuracy survives when the same person’s structural connectome is measured a second time? We drove reservoirs built from diffusion MRI with identical simulated inputs and tested the same frozen linear readout on the original and replacement graphs. The study includes 25 NKI–Rockland and 57 Beijing participants.

## Study results

Independent state noise matched to the size of the repeat-scan displacement overestimated linear recall on the measured repeat scan in both cohorts. The comparison holds the input, decoder weights and cohort transfer penalty fixed. R² measures recall of ten previous simulated inputs. [Full analysis](results/portability-path/README.md) · [Estimates and paired intervals](results/portability-path/shift_summary.csv)

| First visit to second visit | NKI | Beijing |
|---|---:|---:|
| Mean state displacement, % of source training RMS | 132.3% | 56.9% |
| Source R² | 0.191 | 0.237 |
| Expected R² under size-matched isotropic noise | 0.181 | 0.222 |
| R² on the measured repeat scan | 0.043 | 0.153 |
| Additional loss on the repeat scan, paired 95% interval | 0.138 [0.092, 0.192] | 0.069 [0.053, 0.088] |

When 2,352 additional **unlabeled target states** were available for parcelwise alignment, Beijing repeat-scan R² rose to 0.209 and NKI repeat-scan R² reached 0.051. The NKI interval for the alignment gain included zero. [Alignment results and protocol](results/portability-baseline/README.md)

| Question | Where to look |
|---|---|
| How do a numerical rerun, the opposite direction half, a repeat visit and a stranger compare? | [Trajectory agreement](results/portability-path/trajectory_summary.csv) |
| What happens when the decoder penalty or recall task changes? | [Accuracy and portability curves](results/portability-path/path_summary.csv) · [Task comparison](results/pipeline-robustness/task_comparison_summary.csv) |
| Where are the individual comparisons, figures and sensitivity analyses? | [Study results index](results/FINAL_RESULTS_SUMMARY.md) |

## Try the benchmark without participant data

From the repository root in a Python 3.10 environment, run

```sh
python -m pip install -r requirements-benchmark.txt
python -m benchmark.example demo.npz
python -m benchmark demo.npz demo-results --penalties 0.01 1 10
python -m unittest discover -s benchmark/tests -v
```

The example generates two visits for five fictional people. Fixture generation and the first recall run took 2.51 seconds on the tested Windows system with dependencies installed. Open `demo-results/targets.csv` for own versus stranger scores and `demo-results/summary.csv` for participant averages. The same command accepts aligned matrices from another repeat-scan dataset. [Input format, commands and output definitions](benchmark/README.md) · [Timing and verification](benchmark/validation.json)

## Reproduce the study

The [data guide](docs/DATA.md) identifies the original public NKI and Beijing sources and the exact inputs expected by the replay. The [reproduction guide](docs/REPRODUCIBILITY.md) gives the commands for a verified local data tree or a fresh download and explains how to check the numerical outputs. The published-source participant identifiers in the study tables support paired reanalysis.

## Repository layout

| Path | Contents |
|---|---|
| `benchmark/` | Reusable analysis code, command-line entry points, pinned study environment and validation record |
| `benchmark/tests/` | Tests for the reusable analysis |
| `docs/` | Data access and study reproduction guides |
| `scripts/` | Study-specific analysis, data preparation and verification programs |
| `results/` | Study tables, figures, protocols and numerical checks |
| `data/` | Source provenance and file manifests |
| `vendor/` | conn2res simulator files, upstream license and version record |

The source providers and their data-use terms are listed with the numerical environment and simulator provenance. [Sources and software](benchmark/SOURCES.md) · [File checksums](manifest-sha256.json)
