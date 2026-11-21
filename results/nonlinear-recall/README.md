# Three-lag product recall across repeat connectomes

The target is the product of the preceding three uniform input samples. The comparison uses the first and second visits of 25 NKI participants (odd directions) and 57 Beijing participants (whole scan). All arms received the same four fresh target input streams. Each fitted decoder used 2,352 training steps. The pooled decoder distributed that total across the other participants. Only the target-local reference used labeled steps from the evaluated scan for fitting and validation.

## Protocol results

| Mean held-out R² or paired difference | NKI | Beijing |
|---|---:|---:|
| Target-local reference | 0.8592 | 0.9062 |
| Own first scan | −0.00097 | −0.00126 |
| Mean of all foreign decoders | −0.00098 | −0.00103 |
| Anatomically nearest foreign decoder | −0.00102 | −0.00113 |
| Pooled decoder | −0.00124 | −0.00118 |
| Nearest minus mean foreign, 95% interval | −0.000035 [−0.000516, 0.000374] | −0.000102 [−0.000512, 0.000244] |
| Nearest minus pooled, 95% interval | 0.000223 [−0.000579, 0.001078] | 0.000049 [−0.000542, 0.000673] |

Every participant had positive target-local R². The transfer-tuned personal readout scored −0.00097 (NKI) and −0.00127 (Beijing) even on its own original graph under fresh inputs. Thus the frozen-transfer readouts already predicted close to a constant on the source graph. The score of a predictor fixed at zero was −0.00049 and −0.00053 respectively. Ordinary R² assigns exactly zero to the evaluation-sample mean.

The nearest foreign decoder was selected using anatomical correlation with the target connectome before scoring. All 82 choices matched the previous linear analysis, with unique maxima. The mean foreign score averages the 24 or 56 donor scores within each target. The intervals use 10,000 paired resamples of target people, conditional on the fitted donor bank and input signals.

[Locked protocol](../../protocols/NONLINEAR_RECALL_PROTOCOL.md) · [Full estimates and intervals](summary.csv) · [Participant scores](participant_scores.csv) · [All donor comparisons](pair_scores.csv) · [Validation curves](validation_penalties.csv) · [Pooled training and validation assignments](pool_allocations.json) · [Seeds and settings](metadata.json)

## Post hoc regularization diagnostic

The original donor validation chose the grid maximum of 10,000 for all 600 NKI comparisons. In Beijing, 3,004 of 3,192 comparisons selected 1,000. A secondary analysis added quarter-decade spacing from 1 through 10⁸ and the exact source-mean predictor with zero slope. NKI selected that constant predictor in all 600 comparisons. Beijing selected finite penalties from 100 to 3,162, most frequently 316 (2,418 comparisons). The resulting nearest-minus-mean differences were −0.000034 [−0.000514, 0.000375] in NKI and −0.000107 [−0.000484, 0.000219] in Beijing.

Source-local validation on the original grid chose 10⁻¹⁴ for every individual. Those decoders achieved mean R² of 0.894 (NKI) and 0.893 (Beijing) on their own graphs. Applying the same weights to the person's second scan gave negative R² for all 82 people. The medians were −1.15 × 10¹¹ (NKI) and −3.72 × 10¹⁰ (Beijing). Measured in standard deviations of the task target, median prediction spread rose from 0.95 to 339,179 in NKI and from 0.94 to 191,487 in Beijing. This magnitude depends on the weak penalty and the state coordinates. An independent fit using augmented least squares reproduced prediction vectors on both graphs to a maximum relative difference of 3.90 × 10⁻⁸.

[Diagnostic estimates and ranges](boundary_summary.csv) · [Diagnostic participant scores](boundary_people.csv) · [Pairwise selections](boundary_pairs.csv) · [Validation curves](boundary_validation.csv) · [Diagnostic settings](boundary_metadata.json)

The later [matched-penalty analysis](../portability-path/README.md) also tested numerical reruns and direction halves in NKI. The participant median and mean product score stayed below zero for every tested replacement type at every penalty. The [noise diagnostic](../portability-path/noise_source_paths.pdf) describes the sensitivity of the tightly fitted source decoder on unchanged wiring.

## Reproduction

From the project root, run the primary analysis first.

```shell
python scripts/nonlinear_recall.py
python scripts/nonlinear_core_boundary.py
```

The second command verifies every archived primary pair penalty and each corresponding source-self, own-repeat, and foreign score before saving the diagnostic results.
