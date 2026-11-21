# Study results summary

The main study asks how well a linear readout survives substitution of a second measured structural connectome under identical simulated inputs. First-to-second-visit analyses include 25 NKI–Rockland and 57 Beijing participants. Each participant receives equal weight in the cohort means. For the source data behind a particular display, use the [figure and table index](README.md).

## Main comparison

The revised analysis repeats the matched-noise comparison across input maps and graph settings, decomposes the loss using the observed displacement, and compares unlabeled and supervised calibration on equal numbers of target observations. Figures 2 and 3 show the penalty path and parcel alignment. Figure 4 shows how recall changes with calibration budget. [Figure 2](portability-path/Figure_2.pdf) · [Figure 3](portability-path/Figure_3.pdf) · [Figure 4](calibration-budget-curve/Figure_4.pdf) · [Input-map and graph results](headline-robustness/README.md) · [Loss decomposition](reviewer-context/README.md) · [Calibration budgets](calibration-budget-curve/README.md) · [Supplementary tables](supplementary-tables)

Figure 1 first shows the noise-minus-repeat difference across ten input maps and its participant-bootstrap interval for the mean. Its next three panels show each participant’s state displacement and repeat-scan performance under weak and transfer-selected regularization. Rebuild it from the archived participant and cohort tables with `python scripts/plot_manuscript_figure.py`. [Figure 1 PDF](portability-path/Figure_1.pdf) · [Figure 1 PNG](portability-path/Figure_1.png) · [Plotting script](../scripts/plot_manuscript_figure.py)

Across ten signed input maps, size-matched isotropic noise overestimated repeat-scan recall by 0.103 R² in NKI [0.084, 0.122] and 0.065 in Beijing [0.054, 0.078]. The original NKI map produced the largest gap. [All map estimates](headline-robustness/map_portability.csv)

On the original map, at the cohort transfer penalties (10 in NKI, 1 in Beijing), independent state noise matched to the size of each repeat-scan displacement gave higher mean recall than substituting the measured repeat graph. The paired additional loss on the repeat scan was 0.138 R² in NKI (95% interval 0.092 to 0.192) and 0.069 in Beijing (0.053 to 0.088). The expectation under isotropic noise is computed separately for each frozen decoder and displacement. [Analysis and figure](portability-path/README.md) · [Cohort estimates](portability-path/shift_summary.csv) · [Participant estimates](portability-path/shift_people.csv)

| Study question | Figures, tables and verification |
|---|---|
| How much displacement or pattern agreement is observed on technical repeats, a participant’s repeat scan and all other people’s scans? | [Cohort means](portability-path/trajectory_summary.csv) · [Own-minus-stranger intervals](portability-path/trajectory_contrasts.csv) · [Individual pairs](portability-path/trajectory_pairs.csv.gz) · [Definitions and checks](portability-path/trajectory_checks.json) |
| How does source accuracy trade off against transfer as the ridge penalty changes? | [Matched penalty paths](portability-path/accuracy_portability_detail.pdf) · [All score ranges](portability-path/accuracy_portability_full.pdf) · [Distributions and intervals](portability-path/path_summary.csv) · [Independent verification](portability-path/checks.json) |
| What fraction of source accuracy crosses a scan change? | [Own and stranger shares](portability-path/shares_summary.csv) · [NKI technical and between-visit retention using common sources and inputs](portability-path/matched_retention_summary.csv) |
| Can unlabeled observations of a repeat scan improve its score? | [Results and data budget](portability-baseline/README.md) · [Alignment gains and intervals](portability-baseline/summary.csv) · [Pre-scoring protocol](portability-baseline/protocol.md) |

## Supporting comparisons

| Study question | Figures, tables and verification |
|---|---|
| How do single-person, closest-stranger, average-stranger, pooled and locally fitted decoders compare? | [Two-task comparison](pipeline-robustness/task_comparison_summary.csv) · [Closest donor versus own](portability-path/nearest_vs_own.csv) · [Wiring similarity and transfer](transfer-similarity/README.md) |
| What happens on three-lag product recall? | [Study protocol](../protocols/NONLINEAR_RECALL_PROTOCOL.md) · [Task results](nonlinear-recall/README.md) · [Numerical checks](nonlinear-recall/independent_verification.json) |
| How sensitive are the fitted readouts to small state perturbations and participant outliers? | [Source-noise paths](portability-path/noise_source_paths.pdf) · [Participant medians and fractions above zero](portability-path/robust_participant_paths.pdf) |
| How do weights, thresholding, gain and parcellation affect donor selection? | [Pipeline comparisons and figure](pipeline-robustness/README.md) · [Retuned Beijing estimates](pipeline-robustness/retuned_summary.csv) |
| How repeatable is a score when the readout is independently fitted on each visit? | [Reliability estimates](pipeline-robustness/fitted_score_reliability.csv) · [Definitions and settings](pipeline-robustness/fitted_score_reliability_settings.json) |
| What are the original NKI within-acquisition and perturbation controls? | [Matched direction-half contrasts](nki-split-half/matched_visit_contrasts.csv) · [Numerical rerun results](numerical-repeat/summary.csv) |

The task outcomes are simulated recall scores. The readout-state noise controls match the overall displacement, its magnitude in each region, or its full spatial second moment. The extra loss under the real graph change leaves open which anatomical, acquisition or numerical differences produced the relevant displacement. [Data sources](../docs/DATA.md) · [Reproduction instructions](../docs/REPRODUCIBILITY.md)

Online Resource 1, Section S9 reports the paired loss estimates under regional matching, the visit-code exclusion and the one-map extension to the 200-region cortical atlas. [Regional noise](regional-noise/) · [NKI visit-code sensitivity](nki-visit-sensitivity/) · [Beijing Schaefer-200 repeat](schaefer200-repeat/)
