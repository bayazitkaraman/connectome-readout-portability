# Repeat-scan recall on the Beijing Schaefer-200 atlas

All 57 Beijing participants contributed the same first and second visits as in the native-atlas analysis. The original eight signed input parcels for this cortical atlas were held fixed. The relative ridge penalty for each evaluated person was selected using visit-2 validation errors from the other 56 people. The study's original 19 candidate penalties, recall lags, simulation seeds and held-out signals were used. [Analysis protocol](protocol.md)

Expected R² under size-matched isotropic noise was 0.12413, compared with source R² of 0.12625 and measured repeat R² of 0.05257. The paired excess was 0.07156 R² [0.04612, 0.10248] and was positive in 53 of 57 people. Mean state displacement was 109.6% of source-training RMS. Transfer validation selected a penalty of 100 for 52 people and 10 for five. These estimates describe one input map in the same Beijing participants. Coverage and input sites differ from those of the native atlas.

From the study root, run `python scripts/schaefer200_repeat.py` after preparing the Beijing native and Schaefer-200 inputs in `data/replication_candidates/corr`. Verification uses the Schaefer-200 manifest, `scale_input_sites.csv`, `participant_scores.csv` and `retuned_validation.csv` from `results/pipeline-robustness`. The study's donor-specific penalties are excluded from the selection calculation. [Reproduction guide](../../docs/REPRODUCIBILITY.md)

- `people.csv` gives source, noise and repeat scores for each participant under the selected, fixed native and fixed weak penalties.
- `validation.csv` contains 1,083 observed validation errors covering all people and candidate penalties.
- `summary.csv` contains the estimates and participant-bootstrap intervals for each penalty rule. The main row is `retuned`.
- `summary.json` gives the main estimates, selected penalty counts, input sites, checksum and comparisons with the archived fixed-penalty scores and validation errors.
