# Repeat-scan transfer under alternative input and graph settings

For each participant, the first and repeat graphs received the same four fresh input streams. The difference below is expected test R² after substituting independent isotropic state noise on the first graph minus observed test R² on the repeat graph. The noise has the measured mean squared displacement for that graph pair. Confidence intervals are pointwise percentile intervals from 10,000 paired participant resamples, conditional on the maps, validation fits and test streams.

| Setting | NKI difference [95% interval] | Beijing difference [95% interval] |
| --- | --- | --- |
| Original eight driven regions | 0.138 [0.092, 0.192] | 0.069 [0.053, 0.088] |
| Nine other maps averaged within each person | 0.100 [0.082, 0.116] | 0.065 [0.054, 0.078] |
| Radius 0.70 | 0.093 [0.065, 0.122] | 0.048 [0.038, 0.058] |
| Radius 1.00 | 0.101 [0.074, 0.131] | 0.074 [0.052, 0.099] |
| Log weights | 0.117 [0.074, 0.162] | 0.049 [0.038, 0.060] |
| Binary edges | 0.107 [0.076, 0.141] | 0.065 [0.055, 0.078] |
| Fixed count of strongest edges | 0.137 [0.094, 0.186] | 0.069 [0.053, 0.088] |

The nine new map means ranged from 0.084 to 0.125 in NKI and from 0.037 to 0.112 in Beijing. Each of the nine maps gave a positive mean difference in both cohorts. Averaged across those maps within each participant, expected noise R² exceeded measured repeat R² for all 25 NKI participants and all 57 Beijing participants. These estimates condition on the observed region selections. The mean repeat R² across the nine maps was 0.086 in NKI and 0.124 in Beijing.

NKI selected a relative ridge penalty of 10 in every setting. Beijing selected either 1 or 10 depending on the setting. The fixed native-penalty sensitivity gave a larger nine-map mean excess of 0.177 [0.132, 0.235] for Beijing. The excess depends on the readout's operating point. Several conditions at the lightly regularized penalty give very negative absolute scores for the repeat and noise controls; a positive difference between two such scores has little practical bearing on successful recall.

From the staged study directory with native inputs and archived reference tables, run `python scripts/headline_robustness.py`. The protocol specifies the map generation, streams, penalty selection and scoring. The participant is the resampling unit. Each output row in `people.csv` belongs to one participant, setting and ridge rule. The three rules are participant-excluded transfer validation (`retuned`), the original cohort transfer penalty (`native_fixed`) and a fixed relative penalty of 0.001 (`light`). The bootstrap intervals in `summary.csv` condition on that rule and its choices. The source readouts never see labels from the evaluated repeat before testing.

- `input_maps.csv` gives the eight driven atlas indices and signed weights of each map.
- `validation.csv.gz` records validation MSE for each participant, penalty and setting. Average over all other participants to recover an evaluated participant's ridge choice. `penalties.csv` counts the selected penalties.
- `people.csv` and `summary.csv` give absolute scores, excess loss, and the terms of its exactly verified decomposition. `projected_loss` is the mean of the squared projected shifts divided by output variance. `residual_cross_term` is twice their mean product with the source prediction error divided by output variance. The expected isotropic loss is `noise_loss`. Their combination gives `excess_r2 = projected_loss + residual_cross_term - noise_loss`.
- `checks.json` records both numerical identities and comparisons with the archived results. Original person-level R² values and shift statistics agreed with their archived values to within 9 × 10⁻¹⁴. Repeat R² values in the fixed-penalty graph/radius conditions agreed with the independently implemented earlier donor comparison to within 8 × 10⁻¹⁵.
