# Figures and tables in the paper

For the main result, open the [input-map estimates](headline-robustness/map_portability.csv) and select `setting = mean_ten_maps`. Each cohort has one row. The column `excess_r2` is expected recall under size-matched isotropic noise minus measured repeat-scan recall. A positive value means that noise overestimated portability. The ten maps are averaged within each participant before the cohort mean and 95% interval are calculated. The cohort codes `nki` and `bnu1` refer to NKI–Rockland and Beijing Normal University.

## Main figures

| Figure | Data shown | Plotting code |
| --- | --- | --- |
| [1. Matched noise and measured repeat scans](portability-path/Figure_1.pdf) | Panel **a** uses the [ten input maps and their mean](headline-robustness/map_portability.csv). Panels **b–d** use the [participant scores](portability-path/shift_people.csv) and [cohort summaries](portability-path/shift_summary.csv) on the original map. | [Figure 1 script](../scripts/plot_manuscript_figure.py) |
| [2. Linear recall across 19 ridge penalties](portability-path/Figure_2.pdf) | [Participant-level visit scores](portability-path/visit_people.csv.gz) and [NKI reconstruction pairs](portability-path/rung_pairs.csv.gz). The NKI technical curves start from the odd-direction half of the first visit. | [Figures 2 and 3 script](../scripts/plot_manuscript_extension.py) |
| [3. Alignment using unlabeled target states](portability-path/Figure_3.pdf) | [Participant-level scores](portability-baseline/people.csv) and [paired gains and intervals](portability-baseline/summary.csv) after observing 2,352 unlabeled target states. | [Figures 2 and 3 script](../scripts/plot_manuscript_extension.py) |
| [4. Recall across target calibration budgets](calibration-budget-curve/Figure_4.pdf) | [Scores and gains at each budget](calibration-budget-curve/summary.csv) for the frozen readout, unlabeled alignment and labeled residual correction. | [Figure 4 script](../scripts/plot_calibration_budget.py) |

## Main tables

| Table | What it reports | Source |
| --- | --- | --- |
| 1 | Study design. Each row specifies which target information is available, which readout is applied, and how held-out labels are used for scoring. | [Matched-noise and transfer protocol](headline-robustness/protocol.md) · [Calibration protocol](calibration-budget-curve/protocol.md) · [Source-training protocol](source-augmentation/protocol.md) |
| 2 | Displacement, trajectory correlation and relative amplitude by replacement graph on the original map. | [Cohort means and pair counts](portability-path/trajectory_summary.csv) · [Trajectory analysis](../scripts/describe_trajectory_pairs.py) |
| 3 | Zero-label recall from the participant’s own scan, the average other donor, the closest other donor and the budget-matched pool. | [Decoder scores and paired gains](pipeline-robustness/task_comparison_summary.csv), rows with `task = linear` · [Analysis script](../scripts/summarize_benchmark_tasks.py) |
| 4 | Closest-donor recall and its gain over the average other donor for four Beijing graph representations. | [Representation-specific estimates](pipeline-robustness/retuned_summary.csv), rows with `mode = retuned` and metrics `nearest` and `nearest_minus_mean_other` · [Retuning script](../scripts/pipeline_retuning.py) |

## Supplementary tables

The CSVs give the estimates at greater precision than the printed tables.

| CSV | Contents | Supporting results |
| --- | --- | --- |
| [S1](supplementary-tables/S1.csv) | Adding an own or another person’s scan to the donor pool at zero and eight target labels. | [Full calibration curves](density-controls/observational/budget_curves.csv) |
| [S2](supplementary-tables/S2.csv) | Association between wiring similarity advantage and gain at eight labels, including density and disjoint-donor controls. | [Unadjusted association](anatomy-advantage/statistics.csv) · [Density and donor controls](density-controls/) |
| [S3](supplementary-tables/S3.csv) | Closest-donor recall and gain when changing edge weights, edge count, reservoir radius or readout penalty. | [Graph-setting comparisons](pipeline-robustness/summary.csv) |
| [S4](supplementary-tables/S4.csv) | Fixed and retuned penalties for donor selection across Beijing graph representations. | [Retuning estimates](pipeline-robustness/retuned_summary.csv) |
| [S5](supplementary-tables/S5.csv) | NKI readouts trained on the first-visit reference, its numerical reruns, or both direction halves. | [Participant-level source-training scores](source-augmentation/people.csv) |
| [S6](supplementary-tables/S6.csv) | Noise-minus-repeat recall across ten input maps and five other graph settings. | [Setting-specific results](headline-robustness/summary.csv) · [Map averages](headline-robustness/map_portability.csv) |
| [S7](supplementary-tables/S7.csv) | Frozen, aligned and label-corrected scores at the same target-state budgets. | [Individual calibration scores](calibration-budget-curve/people.csv) |
| [S8](supplementary-tables/S8.csv) | Graph density, donor selection and the share of excess loss captured by the spatial noise control. | [Cohort and donor summaries](reviewer-context/README.md) · [Ten-map spatial shares](headline-robustness/map_portability.csv) |
| [S9](supplementary-tables/S9.csv) | Recall loss under isotropic noise, regional noise, noise matching the full spatial second moment and the measured repeat. | [Participant-by-map scores](regional-noise/participant_map.csv) · [Paired estimates](regional-noise/summary.csv) |

Section S9 also reports two checks alongside the table. [NKI visit-code sensitivity](nki-visit-sensitivity/summary.csv) repeats the ten-map comparison after excluding the uncertain visit code. [Beijing Schaefer-200 results](schaefer200-repeat/summary.csv) give the one-map comparison in the same 57 Beijing participants.

For the fitted-score repeatability analysis, three-lag product recall and further controls, see the [study results summary](FINAL_RESULTS_SUMMARY.md). To regenerate the results from the source matrices, follow the [reproduction guide](../docs/REPRODUCIBILITY.md).
