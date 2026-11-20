# Training on several reconstructions of a first scan

For NKI linear recall, distributing the original 2,352 labeled samples between the two first-visit diffusion-direction halves increased mean frozen repeat R² from 0.04284 to 0.05589. The paired gain was 0.01305 (95% participant bootstrap interval 0.00082 to 0.02761). Training on the odd reference plus its 20 numerical reruns gave a smaller gain with an interval that included zero.

| First-visit training graphs | Labeled times per graph | Original-source R² | Own repeat R² | Paired repeat gain (95% interval) | Participants improving |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odd reference | 2,352 | 0.19098 | 0.04284 | Reference | — |
| Odd reference and 20 numerical reruns | 112 | 0.19058 | 0.04453 | 0.00169 (−0.00006 to 0.00455) | 15/25 |
| Odd and even direction halves | 1,176 | 0.17854 | 0.05589 | 0.01305 (0.00082 to 0.02761) | 15/25 |

Retained share is the ratio of the cohort mean repeat and original-source scores for each model. It was 22.43% for the reference, 23.36% with numerical augmentation, and 31.30% with direction halves. Relative to the reference, the paired differences were 0.93 percentage points (0.04 to 2.37) and 8.87 points (2.39 to 15.79). Direction-half training lowered mean source R² and raised repeat R², so both changes contribute to its retention ratio. Repeat R² was positive for 20, 20 and 19 participants, respectively. The interval for the absolute gain under numerical augmentation includes zero even though the pointwise interval for its ratio gain falls above it.

The source graphs were measured derivatives of the same acquisition. Of the 500 numerical reruns, 246 reproduced all off-diagonal weights of the odd reference. The intervals condition on the observed participants' source variants, synthetic input streams and the specified balanced allocation of training times. They provide pointwise uncertainty for the two pre-scoring comparisons. The Beijing release did not provide these technical variants.

The [protocol](protocol.md) fixes the training budget, absolute ridge penalty and paired contrasts. [Participant scores and training allocations](people.csv) and [cohort estimates and intervals](summary.csv) report both denominators. The value in the original arm's `n_improved` field counts strict improvements over itself.

Run `python scripts/source_augmentation.py` from the benchmark root after preparing the checksum-verified NKI data and the archived shifted-state scores. The calculation checked unaugmented scores against the earlier forward evaluation in all 25 people (maximum absolute discrepancy less than 1.4 × 10⁻¹⁶ R²), recovered the original scores exactly when all input graphs were replaced with copies of the original, and compared all predictions with a separate covariance-form ridge solution (maximum absolute difference 2.1 × 10⁻¹⁴). [Verification and seeds](checks.json)
