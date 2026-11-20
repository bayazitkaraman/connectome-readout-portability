# Zero-label transfer across ridge penalties

The archived single-scan analysis tested 19 relative penalties from 10⁻¹⁴ to 10⁴ in factors of ten. All 600 primary NKI comparisons selected 10. All 3,192 Beijing comparisons selected 1. Both choices were inside the archived grid. An independent simulation reproduced the selected penalties and held-out own and stranger R² scores for all 10,392 comparisons, including the additional NKI half and visit combinations. The largest absolute score discrepancy was 6.22 × 10⁻¹⁵.

The sensitivity run tested 177 penalties from 10⁻¹⁶ to 10⁶, with eight steps per decade. For each target and stranger, tuning used the other 23 NKI or 55 Beijing participants. We minimized the mean squared error for transfer from each tuning participant’s source scan to their own target scan on separate validation streams. Both compared decoders used that penalty. The reported R² averages ten delay-specific held-out scores, with 2,048 fresh simulated steps per target. Reported 95% intervals resample participants 10,000 times, holding the donor bank and its tuning estimates fixed.

| Visit 1 to visit 2 | Mean own-scan R², archived → refined | Mean stranger R², refined | Mean own minus stranger, refined (95% interval) | Refined penalty range |
|---|---:|---:|---:|---:|
| NKI, odd directions, 25 people | 0.04284 → 0.04109 | 0.02924 | 0.01185 (−0.02611 to 0.04419) | 10 to 23.71 |
| Beijing, 57 people | 0.15265 → 0.15503 | 0.11680 | 0.03823 (0.02135 to 0.05310) | 1 to 1.33 |

The Beijing own-scan advantage falls from 0.0492 to 0.0382 and remains positive. The NKI mean gain falls from 0.0191 to 0.0119, with the refined interval spanning zero. Within-visit values are more sensitive to penalty resolution. For example, NKI visit 1 odd-to-even mean own R² increases from 0.1190 to 0.1683. All four refined same-visit comparisons show higher mean scores for own halves than strangers and higher absolute own scores than the corresponding cross-visit comparisons.

The [condition summaries](grid_summary.csv), [participant means](grid_people.csv), [all pair scores and penalties](grid_pairs.csv), [validation curves](grid_validation.csv), [replay metadata](grid_metadata.json), and [analysis code](../../scripts/transfer_grid_audit.py) accompany this note. The validation curves average across the full cohort for display. The comparisons reported above used the identity exclusions described above.
