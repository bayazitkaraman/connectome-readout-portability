# Study analysis scripts

The paper asks whether noise on a first-scan reservoir predicts how a frozen decoder will perform on a measured repeat. The scripts below reproduce the comparisons in the paper. To run them together from the source matrices, use the study replay. It stages the inputs, runs the scripts in dependency order and checks the regenerated results. [Reproduction guide](../docs/REPRODUCIBILITY.md) · [Paper figures and tables](../results/README.md)

## Main comparison and sensitivity analyses

| Question | Analysis code | Output to inspect |
| --- | --- | --- |
| How much does isotropic noise overestimate repeat recall across ten input maps? | [Fit and score each map and graph setting](headline_robustness.py) · [Average maps within participants](summarize_map_portability.py) | [Map and cohort estimates](../results/headline-robustness/map_portability.csv) |
| What changes when noise is matched to each region separately? | [Regional noise and spatial loss decomposition](regional_noise.py) | [Participant-by-map results](../results/regional-noise/participant_map.csv) · [Paired intervals](../results/regional-noise/summary.csv) |
| How sensitive is the NKI result to the uncertain second-visit label? | [Exclude the participant from evaluation and tuning](nki_visit_sensitivity.py) | [Full and restricted samples](../results/nki-visit-sensitivity/summary.csv) |
| Does the isotropic overestimate persist with a different atlas? | [Score one Schaefer-200 input map in Beijing](schaefer200_repeat.py) | [Alternate-atlas estimate](../results/schaefer200-repeat/summary.csv) |
| What does changing the readout penalty do? | [Between-visit and other-person comparisons](run_visit_portability_path.py) · [Numerical reruns and direction halves](run_rung_portability_path.py) | [Source and target scores along the penalty path](../results/portability-path/path_summary.csv) |
| Can unlabeled target states, labeled target states, or a second source reconstruction recover recall? | [Parcelwise alignment](unlabeled_parcel_alignment.py) · [Label budget](calibration_budget_curve.py) · [First-visit training](source_augmentation.py) | [Alignment](../results/portability-baseline/summary.csv) · [Calibration](../results/calibration-budget-curve/summary.csv) · [Source training](../results/source-augmentation/summary.csv) |

## Figures

The plotting scripts read the archived score tables. They can regenerate the four paper figures without downloading connectomes. Run these commands in a working copy with the study environment installed. They write the PDF and PNG figures under `results/` in that copy.

```sh
python scripts/plot_manuscript_figure.py
python scripts/plot_manuscript_extension.py
python scripts/plot_calibration_budget.py
```

The first command draws Figure 1, the second Figures 2 and 3, and the third Figure 4. The results index identifies the plotted data for each figure. [Figure and table index](../results/README.md)

## Reuse with another cohort

For a new dataset, use the `benchmark` command with aligned matrices. The `--state-comparisons` option returns frozen same-person source and repeat scores, expected isotropic and spatially matched scores, and the effect of parcelwise alignment. The regional-noise analysis used in Table S9 has its own study script. [Input schema and output columns](../benchmark/README.md) · [Regional-noise calculation](regional_noise.py)
