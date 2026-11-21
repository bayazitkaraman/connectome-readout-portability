# Sensitivity to the unmatched NKI visit code

For A00034854, the NKI archive identifies visits DS2 and NFB2R. The public crosswalk maps DS2 to BAS1 and has no entry mapping this participant's NFB2R. The participant with an unusually low edge-weight correlation, A00063383, has two mapped visits. The odd-half correlation between visits was 0.7646 for A00034854 and 0.0014 for A00063383.

This analysis verifies each of the 100 deterministic reference graphs against the source HDF and each of the 50 visit records against the public crosswalk. It repeats the ten-map linear-recall contrast after removing A00034854 from evaluation and from each remaining participant's penalty selection. Each of the 24 remaining participants then has 23 independent participants on the validation panel. All 240 selected penalties agreed with the original choices, permitting the archived scores to be reused.

| First-to-second visit sample | People | Expected isotropic R² minus measured repeat R² | Paired 95% interval |
|---|---:|---:|---:|
| All participants | 25 | 0.10339 | 0.08417 to 0.12187 |
| Both visits mapped | 24 | 0.10551 | 0.08583 to 0.12422 |

Each participant contributes one ten-map average to the estimate. The interval uses 10,000 participant resamples conditional on the maps, signals and validation panel.

Run `python scripts/nki_visit_sensitivity.py` from the study root after preparing the NKI inputs and native headline outputs. The script reads the NKI HDF, prepared NPZ, provenance and visit crosswalk from `data/nki`, plus `people.csv`, `validation.csv.gz` and `map_portability.csv` from `results/headline-robustness`. [Reproduction guide](../../docs/REPRODUCIBILITY.md)

- `source_visits.csv` records source identifiers, mapped visit labels, matching graph counts and odd-half edge correlations.
- `participant_map.csv` gives each person's outcomes and the original and reduced-panel penalty choices. The excluded participant has an empty reduced-panel penalty.
- `participant_means.csv`, `map_summary.csv` and `summary.csv` give individual ten-map averages, per-map group summaries and the two ten-map group estimates.
- `verification.json` records the source and prepared file hashes, graph checks, validation decisions and agreement with the original full-sample results.
