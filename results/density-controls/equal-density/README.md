# Refit after holding the number of positive edges fixed

Within each cohort, retain the strongest strictly positive undirected edges in every first and second scan, using the minimum observed edge count as the common budget. This retains 177 of 3,403 possible edges per NKI graph and 1,291 per Beijing graph. Keep every participant, retain the original weights on surviving edges, and rescale the resulting reservoir to spectral radius 0.90. Rebuild all common-bank and candidate readouts with the original input sites, stimulation, validation, donor assignments, label allocations, and six calibration draws. Ties at the retention boundary are resolved by a single, scan-independent random permutation of atlas edges per cohort, first with seed 61973 and again with 61974. The two seeds and these choices were fixed before scoring. The cut split a tie in 31 of 50 NKI scans and 107 of 114 Beijing scans.

| Outcome at eight target examples, equal donor share | NKI native | NKI fixed count | Beijing native | Beijing fixed count |
|---|---:|---:|---:|---:|
| Mean gain from own vs another person's scan (R²) | +0.00530 | +0.00598 | +0.00176 | +0.00176 |
| Across-person similarity/gain Spearman correlation | 0.720 | 0.736 | 0.151 | 0.156 |
| Paired participant-bootstrap correlation interval | 0.384 to 0.920 | 0.450 to 0.922 | −0.133 to 0.430 | −0.127 to 0.432 |
| Target-only mean R² at 2,352 examples | 0.9522 | 0.9545 | 0.8362 | 0.8361 |

After equalizing edge counts, the primary association persisted in NKI and remained weak in Beijing. Under a second ordering of the boundary ties, eight-example correlations were 0.745 and 0.156, respectively. With 25% of source examples from the added scan, the first ordering yielded correlations of 0.652 in NKI and 0.247 in Beijing. The participant resampling intervals in this sensitivity keep the donor assignments and streams fixed.

Thresholding removes low-weight edges and can change network topology. Median total original streamline weight retained was 90.72% in NKI and 99.96% in Beijing. In the primary tie ordering the number of active readouts was 43 to 74 in NKI and 64 to 75 in Beijing, out of 75 possible, on the identical standard source signal. Every graph retained measurable activity. Equalizing counts does not equalize regional degree, weight composition, overlap across scans, or the effects of reconstruction quality.

Each cohort and tie-seed folder contains graph checks, pooled scores at zero and eight labels, paired results, full target-only recovery, and the summary estimates. For one original target/donor pair per cohort and tie seed, all scores from the cached-target implementation were verified against the original pipeline with an absolute and relative tolerance of 1e−10. Run `python scripts/equal_density_sensitivity.py` from the project root to regenerate.
