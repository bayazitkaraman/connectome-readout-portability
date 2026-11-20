# Reproduce the repeat-connectome study

The reusable benchmark runs a fixture with five fictional participants. The study replay uses the original NKI–Rockland and Beijing matrices and regenerates the main paper analyses in a separate directory. Run all commands below from the repository root. [Synthetic quick start](../README.md) · [Input sources](DATA.md)

## Set up the study environment

Use Python 3.10.19 to match the verified environment. Create a virtual environment with `python -m venv .venv`, then activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux.

```sh
python -m pip install -r benchmark/requirements-study.txt
python -m unittest discover -s benchmark/tests -v
```

## Replay the analyses

Choose a new working directory name. To retrieve the study inputs from their providers and reproduce the tables, run

```sh
python -m benchmark.reproduce replay --fetch
```

For an existing local collection of verified study inputs, run the following command using the directory containing `nki/` and `replication_candidates/`.

```sh
python -m benchmark.reproduce replay --data-root /path/to/data
```

The command checks the inputs, copies the analysis programs and writes estimates under `replay/results/`. The replay covers linear and product recall, the source–target penalty paths, fixed-graph noise, NKI numerical reruns and direction halves, the donor and pipeline comparisons, trajectory agreement and the unlabeled alignment baseline. It fits the budget-matched linear pooled decoder independently and compares it to shipped calibration scores. The earlier calibration tables serve as references for that comparison. [Replay implementation](../benchmark/reproduce.py)

## Check the results

On success, `replay/replay_verification.json` reports numerical differences from the shipped study tables. The release validation used verified local data. Its record includes the synthetic example, independent checks against original donor-pair scores, and replay discrepancies. [Validation record](../benchmark/validation.json) · [Study results index](../results/FINAL_RESULTS_SUMMARY.md)

To verify the reusable CLI against the native-resolution linear study, run

```sh
python -m benchmark replay/data/nki/nki_reference.npz nki-results --half odd
python -m benchmark.verify_archive nki-results nki
python -m benchmark replay/data/replication_candidates/corr/bnu1_dkt_retest.npz beijing-results
python -m benchmark.verify_archive beijing-results bnu1
```

The archive comparison matched all 600 forward stranger pairs in NKI and all 3,192 in Beijing, with maximum absolute R² differences below 2.4 × 10⁻¹⁵. The study and generic CLI use identical input streams for both graphs of a source–target comparison. [Generic analysis and matrix conventions](../benchmark/README.md)

## Versioned archive

The repository includes a metadata template for a later public release. The creator list, project license and version await confirmation by the authors. When the public version is archived, its tag and version DOI can identify the exact study files. [Draft release metadata](../benchmark/zenodo-metadata.example.json)
