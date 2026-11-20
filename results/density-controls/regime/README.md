# Eight-example sensitivity to reservoir gain and source regularization

On the original MRI-derived networks, rerun the equal-donor first-scan swap at eight labeled target examples. Use spectral radii 0.70, 0.90 and 1.00. At each radius, select the source ridge penalty using the usual common-donor training/validation split. Fit both candidate models using 0.1, 1 or 10 times the selected *relative* penalty for that target/comparator pair. All nine combinations per cohort use the same three comparison people per target, six calibration draws, source and target training budgets, atlas input mapping and target-residual ridge selection. These settings were chosen before the sweep. The primary study analysis remains at radius 0.90 and multiplier 1.

## Cross-validated source penalties

The rank correlation compares the participant's anatomical similarity advantage and its accuracy advantage from adding their personal scan. The average accuracy gain is the average difference in held-out R², personal minus other. Intervals in the accompanying tables resample target participants, hold donor assignments and simulation streams fixed, and apply separately to each setting.

| Cohort | Radius | Mean gain (R²) | Similarity/gain correlation | Paired 95% interval for correlation |
|---|---:|---:|---:|---:|
| NKI-Rockland | 0.70 | +0.00341 | 0.735 | 0.438 to 0.908 |
| NKI-Rockland | 0.90 | +0.00530 | 0.720 | 0.384 to 0.920 |
| NKI-Rockland | 1.00 | +0.00704 | 0.652 | 0.300 to 0.877 |
| Beijing | 0.70 | +0.00315 | 0.269 | 0.007 to 0.501 |
| Beijing | 0.90 | +0.00176 | 0.151 | −0.133 to 0.430 |
| Beijing | 1.00 | +0.00134 | 0.057 | −0.212 to 0.317 |

With both the radius and source ridge varied, correlations ranged from 0.314 to 0.735 in NKI and from 0.057 to 0.269 in Beijing. The mean accuracy gain spanned −0.00002 to +0.01695 in NKI and +0.00028 to +0.00712 in Beijing. At the original setting, comparisons against the previously saved per-participant, per-comparator, per-draw R² values differed by less than 1e−16 in both cohorts. The exercise measures accuracy at eight labels and does not estimate label savings.

[All NKI settings](nki/summary.csv) · [All Beijing settings](bnu1/summary.csv). Each cohort directory also includes its validation-selected source penalties and raw scores. Run `python scripts/scan_swap_regime_sensitivity.py` from the project root.
