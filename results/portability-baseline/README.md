# Unlabeled parcel alignment before repeat-scan recall

Matching the mean and standard deviation of each target readout region to its source-training values raised mean Beijing repeat-scan R² from 0.153 to 0.209. The change was smaller and uncertain in NKI, where mean R² rose from 0.043 to 0.051. Both comparisons use the same frozen linear decoder and the same four held-out input streams before and after alignment. [Cohort results](summary.csv)

| First to second visit | NKI (25 people) | Beijing (57 people) |
|---|---:|---:|
| Transfer-selected ridge penalty | 10 | 1 |
| Source R² | 0.191 | 0.237 |
| Untouched repeat R² | 0.043 | 0.153 |
| Aligned repeat R² | 0.051 | 0.209 |
| Paired alignment gain in R² [95% interval] | +0.009 [−0.017, +0.040] | +0.056 [+0.041, +0.074] |
| Source R² retained, untouched → aligned | 22.4% → 26.9% | 64.3% → 87.9% |
| People with higher R² after alignment | 14/25 | 54/57 |
| People above zero, untouched → aligned | 20/25 → 16/25 | 53/57 → 57/57 |

Alignment requires an additional 2,352 unlabeled observations of the 75 readout regions on the target graph. These observations came from a separate simulated input stream. Each target parcel’s calibration mean and population SD were used to map held-out states to the corresponding source training mean and SD. A parcel constant in either reference series was set to its source training mean. Target recall labels entered only the evaluation. The untouched comparison had no target calibration observations. The procedure tests the practical value of coordinatewise alignment at this budget. The underlying causes of scan differences remain open. [Protocol recorded before alignment was scored](protocol.md)

The intervals resample participants, with observations paired before and after alignment, conditional on the simulated calibration and evaluation streams. Retention divides the cohort mean repeat R² by the cohort mean source R². The frozen source and untouched repeat scores for all 82 people reproduced the archived values within 1.1 × 10⁻¹⁵ R². [Participant results](people.csv) · [Verification](checks.json)

From the repository root with the two cohort inputs and the original visit-pair result available, run `python scripts/unlabeled_parcel_alignment.py`.
