# Reproduce the repeat-connectome study

The study replay starts from the NKI and Beijing structural connectomes and regenerates the reported analyses in a new directory. To inspect the numbers immediately, open the archived tables. To try the method on fictional participants, use the short example on the repository home page. [Figure and table index](../results/README.md) · [Synthetic example](../README.md#run-the-synthetic-example)

## Set up Python

The verified interpreter is Python 3.10.19. From the repository root, create an environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then install the study dependencies and run the tests.

```sh
python -m pip install -r benchmark/requirements-study.txt
python -m unittest discover -s benchmark/tests -v
```

## Run the full replay

Choose a new output directory. To fetch the study inputs from the providers and run the analyses, use

```sh
python -m benchmark.reproduce replay --fetch
```

If you already have the verified inputs, use this command instead. Replace `/path/to/data` with the directory containing `nki/` and `replication_candidates/`. The data guide lists the necessary files and their sources. [Input files and checksums](DATA.md)

```sh
python -m benchmark.reproduce replay --data-root /path/to/data
```

Both routes check the graph inputs, fit and score the readouts, and write outputs under `replay/results/`. The replay includes the ten input maps, the regional-noise control, the NKI visit-label sensitivity, the one-map Beijing Schaefer-200 comparison and the four manuscript figures. It also reproduces the penalty paths, nonlinear recall, technical comparisons, alignment and calibration analyses. The earlier calibration scores are supplied as references for the budget-matched linear pooled decoder. [Scripts by study question](../scripts/README.md) · [Exact sequence of analysis calls](../benchmark/reproduce.py)

## Check the output

A successful run writes `replay/replay_verification.json`, which compares regenerated summary columns with the shipped tables. In the verified run, all 33 tables passed, the largest absolute difference across 125 checked columns was below 1.4 × 10⁻¹², and all four manuscript PNGs matched pixel for pixel. The public-source input route was separately checked against the NKI source and all 342 Beijing edge lists at three atlas scales. [Validation record](../benchmark/validation.json) · [Numerical results](../results/FINAL_RESULTS_SUMMARY.md)

For a separate test of the general-purpose command, use the prepared native matrices from the replay. The verification commands compare forward other-person transfer scores with the study archive.

```sh
python -m benchmark replay/data/nki/nki_reference.npz nki-results --half odd --state-comparisons
python -m benchmark.verify_archive nki-results nki
python -m benchmark replay/data/replication_candidates/corr/bnu1_dkt_retest.npz beijing-results --state-comparisons
python -m benchmark.verify_archive beijing-results bnu1
```

The fresh-install check covered all 600 NKI and 3,192 Beijing forward other-person scores. Both source and target graphs received the same test inputs. The full replay also compares the state scores against the separate study implementation for the 82 forward same-person comparisons. [Benchmark output definitions](../benchmark/README.md) · [State comparison check](../scripts/verify_cli_state_comparisons.py)

## Cohort age summary

To regenerate the NKI age summary, obtain the public Rockland imaging index from the linked sources. From the prepared replay directory, run `python scripts/reviewer_context.py --nki-age-index /path/to/aws_links.csv`. The script checks the provider hash, matches the baseline diffusion and T1 ages, and exports cohort aggregates. [Age sources and checksum](../results/density-controls/age_sources.md) · [Cohort and donor summaries](../results/reviewer-context/README.md)
