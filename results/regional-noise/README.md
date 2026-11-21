# Noise matched to regional displacement power

This retrospective analysis uses the first-to-second visit comparison for linear recall in all 25 NKI and 57 Beijing participants. Each of the ten eight-region input maps uses its original native graph, radius of 0.90, simulated signals and participant-excluded transfer penalty.

For a given scan pair, let D contain the repeat states minus the source states at the 2,048 scored time points, and let M = DᵀD/T. The regional control applies independent, zero-mean observation noise with variance Mⱼⱼ in parcel j. For output weights w, its expected increase in mean squared error is the sum of wⱼ²Mⱼⱼ. We divide by each delayed target's test variance before averaging the ten outputs. The isotropic control distributes the same total power equally across parcels. The spatial control preserves the entire matrix M. All three expectations use the measured scan pair.

| Mean loss in R² across ten maps | NKI | Beijing |
|---|---:|---:|
| Isotropic noise | 0.00692 | 0.00901 |
| Independent regional noise | 0.03184 | 0.04205 |
| Full spatial second moment | 0.05426 | 0.06058 |
| Measured repeat | 0.11031 | 0.07449 |
| Measured minus regional-noise loss, with 95% interval | 0.07847 [0.06247, 0.09333] | 0.03245 [0.02567, 0.03983] |

Across ten maps, regional differences in power accounted for 24.1% [20.5%, 28.6%] of the mean excess over isotropic noise in NKI and 50.4% [44.4%, 57.0%] in Beijing. Off-diagonal second moments accounted for another 21.7% [19.8%, 23.6%] and 28.3% [20.1%, 36.8%]. On Beijing map 13, regional-noise loss exceeded measured repeat loss by 0.03055 R². Ten-map summaries average within person before averaging people. Each pointwise interval uses 10,000 paired participant-bootstrap resamples, with the maps, penalties and input streams fixed. Shares are ratios of cohort means.

From the study root, prepare the native inputs and headline results, then run `python scripts/regional_noise.py`. The script reads the selected penalties and reference scores from `results/headline-robustness/people.csv`. [Reproduction guide](../../docs/REPRODUCIBILITY.md)

- `participant_map.csv` contains all 820 participant-map rows, including their selected penalties.
- `summary.csv` gives cohort estimates and intervals for each map and for the ten-map average.
- `validation.json` records agreement with the original calculations. The script independently checks each off-diagonal contribution by direct multiplication.
