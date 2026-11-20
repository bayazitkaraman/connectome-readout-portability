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

On success, `replay/replay_verification.json` reports numerical differences from the shipped study tables. The study analyses were checked using checksum-verified local inputs. The input-preparation route was also tested in an empty directory against the original public providers, including all 342 Beijing edge lists at the three atlas scales. The validation record covers the synthetic example, native-cohort comparisons and numerical replays. [Validation record](../benchmark/validation.json) · [Study results index](../results/FINAL_RESULTS_SUMMARY.md)

To verify the reusable CLI against the native-resolution linear study, run

```sh
python -m benchmark replay/data/nki/nki_reference.npz nki-results --half odd
python -m benchmark.verify_archive nki-results nki
python -m benchmark replay/data/replication_candidates/corr/bnu1_dkt_retest.npz beijing-results
python -m benchmark.verify_archive beijing-results bnu1
```

The archive comparison matched all 600 forward stranger pairs in NKI and all 3,192 in Beijing, with maximum absolute R² differences below 2.4 × 10⁻¹⁵. The study and generic CLI use identical input streams for both graphs of a source–target comparison. [Generic analysis and matrix conventions](../benchmark/README.md)

## Additional comparisons in version 0.2.0

The full replay also fits the source-augmentation strategies, estimates retention intervals, compares graph settings and input maps, runs the target-budget analysis and regenerates the four main figures. It invokes the reusable CLI with `--state-comparisons` for both native cohorts and compares every forward participant's displacement, matched-noise and alignment scores to the independent study implementation. The archived CLI outputs include both visit directions. [Scoring comparison](../scripts/verify_cli_state_comparisons.py) · [Input-map and graph protocol](../results/headline-robustness/protocol.md) · [Budget protocol](../results/calibration-budget-curve/protocol.md)

To reproduce the descriptive ages, obtain the official public imaging index from the data guide. In the prepared replay directory run `python scripts/reviewer_context.py --nki-age-index /path/to/aws_links.csv`. The script verifies the provider hash and cross-checks baseline diffusion and T1 ages. Only cohort aggregates are exported. [Age sources and checksum](../results/density-controls/age_sources.md) · [Cohort and donor checks](../results/reviewer-context/README.md)

Version 0.2.1 adds `python scripts/summarize_map_portability.py` after the native CLI evaluations. The script uses the archived participant and penalty tables to reproduce the ten-map estimate, retention and spatial shares, the original-map rankings and the reverse-visit check. It writes `results/headline-robustness/map_portability.csv` and `reporting_checks.json`.

The snapshot carries versioned creator, citation and license metadata. The distributed manifest identifies the bytes in this version. [Citation](../CITATION.cff) · [Metadata](../.zenodo.json) · [Checksums](../manifest-sha256.json)
