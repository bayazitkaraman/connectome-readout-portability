# Wiring similarity and zero-label decoder transfer

In two repeat-scan cohorts, choosing another person's decoder by the similarity of their structural connectivity to the target improved recall of a simulated random signal. Among cross-visit cases for which strangers offered comparably similar wiring, the estimated additional benefit of a personal source was close to zero, with appreciable uncertainty. In the NKI within-acquisition data, comparable strangers were unavailable for every target. These follow-up analyses were motivated by inspection of earlier results.

## Choosing a donor before observing any target labels

We tested the original first-to-second scan direction. For each target, we selected the other person with the largest Pearson correlation of upper-triangle structural edge weights, including zeros. We applied that person's frozen decoder to the target's held-out signals. The comparison is the mean of the other people's individually fitted decoder scores on those same signals. Every transfer decoder used the original penalty chosen on identities outside the target and donor pair. The choice used the target wiring and zero target recall labels.

| Cohort | Mean other R² | Closest other R² | Selection gain [paired 95% interval] | Targets improved | Own R² |
|---|---:|---:|---:|---:|---:|
| NKI, 25 people | 0.024 | 0.114 | +0.090 [0.068, 0.113] | 24/25 | 0.043 |
| Beijing, 57 people | 0.103 | 0.167 | +0.063 [0.048, 0.079] | 52/57 | 0.153 |

Some donors were selected repeatedly. As a check, for each held-out person we estimated how often each donor was the closest match to the other targets, excluding the held-out person from both roles. Selecting a donor with those frequencies gave R² of 0.044 in NKI and 0.119 in Beijing. Matching to the actual target's wiring increased R² by 0.070 [0.055, 0.085] and 0.048 [0.034, 0.062] over that comparison. With finer ridge selection, the closest donor still beat the average donor by 0.070 in NKI and 0.048 in Beijing. All intervals for these comparisons are participant bootstraps conditional on the observed bank of fitted models and test inputs. [Original penalty results](nearest_donor_summary.json) · [Finer grid](nearest_donor_dense_summary.json) · [Participant scores](nearest_donor_people.csv)

## How much does a decoder lose on another scan?

We replayed identical fresh signals on the source and destination wiring matrices using the exact same frozen readout weights and intercept. Each personal decoder had 2,352 simulated fitting labels from the source scan. An additional source-tuned comparison used 1,176 independent source-validation labels. The transfer penalty used validation from other people. No target outcomes were used to fit or choose a decoder.

| First to second visit | Locally tuned, scored on source | Transfer-tuned, scored on source | Same frozen decoder on repeat | Source minus repeat [95% interval] |
|---|---:|---:|---:|---:|
| NKI | 0.965 | 0.191 | 0.043 | 0.148 [0.099, 0.205] |
| Beijing | 0.831 | 0.237 | 0.153 | 0.085 [0.066, 0.106] |

The larger part of the gap from a locally tuned fit to zero-label transfer appeared when regularization changed on the source graph. A further loss occurred when only the simulated wiring changed. For NKI's same-acquisition odd-to-even comparison at the fixed penalty of 10, the source-to-target changes were 0.190 to 0.153 and 0.190 to 0.176. These correspond to about 80% and 93% of the same frozen decoder's source R² (ratios of cohort means). The earlier “roughly a fifth” comparison used a separately tuned high-accuracy local fit as its denominator. The two comparisons answer different questions. [Source scores and uncertainty](source_self_reference.md)

On the **identical even-direction target**, changing the odd-direction source from the same visit to the other visit lowered R² from 0.153 to 0.059 at visit 1 (25/25 people) and from 0.176 to 0.048 at visit 2 (24/25). The mean paired advantages were 0.094 [0.063, 0.127] and 0.129 [0.085, 0.177]. This is the follow-up comparison at a common penalty. The earlier condition-specific tuning remains documented. Only NKI supplied 64 odd and 64 even directions per visit. The halves shared nine baseline images and full-acquisition preprocessing, and independent visits add multiple sources of variation. [Fixed-penalty checks](similarity_penalty_checks.json) · [NKI release](https://zenodo.org/records/4041549)

## How far does the similarity explanation reach?

All six reported NKI similarity-bin counts and rounded scores reproduced at a fixed penalty of 10 across the 12 directed conditions. Mean stranger R² rose from 0.021 at correlation below 0.45 to 0.143 at 0.75–0.85. No stranger reached 0.85. The six personal cross-visit observations in the 0.85–0.92 bin came from one person. In Beijing, 0.85–0.92 included 33 personal repeat pairs (R² 0.170) and 223 stranger pairs (0.167). Beijing's middle half of personal repeat similarities spanned 0.841–0.881. Similar bin means are descriptive because different people can supply the observations in each bin. [NKI bins and participant counts](similarity_bins_nki.csv) · [Beijing bins](similarity_bins_bnu1.csv)

For each target and directed scan combination, we identified the two strangers immediately below and above its personal similarity and interpolated their scores to the personal similarity. The own-minus-interpolated difference averaged −0.004 R² in NKI [−0.020, 0.012], with overlap in 107 of 200 cross-visit conditions across 16 of 25 people. In Beijing it was −0.003 [−0.032, 0.021] across 33 of 57 people. We averaged repeats within each person before summarizing, with fixed observed donors. In the 100 NKI within-acquisition conditions no stranger came within 0.05 of the personal similarity for that target. An independent identity or session effect at that high similarity remains unresolved. [NKI matched comparisons](similarity_matches_nki.csv) · [Beijing matched comparisons](similarity_matches_bnu1.csv)

The reported −0.007 coefficient reproduces for a regression on the 300 *personal* rows, with wiring similarity and a same-acquisition flag. Its 95% interval was −0.032 to 0.019 by participant bootstrap and −0.034 to 0.020 using a target-clustered t interval. Identity is constant across these rows. In the linear model including strangers, the similarity-adjusted personal coefficient was −0.002 and the extra same-acquisition coefficient was +0.017. Inference in the highest similarity range depends on extrapolation. [Regression results and dependence checks](similarity_regression_nki.csv)

## Penalty and participant sensitivity

The original grid contained 19 penalties from 10⁻¹⁴ to 10⁴. All 600 primary NKI comparisons selected 10, and all 3,192 Beijing comparisons selected 1. An expanded 177-point grid retained the conclusions, with mean cross-visit personal gains over an average stranger of +0.012 [−0.026, 0.044] in NKI and +0.038 [0.021, 0.053] in Beijing. These were +0.019 and +0.049 on the original grid. [Grid audit](grid_audit.md)

Leaving out A00063383 as a target changed the original NKI personal gain from 0.0191 to 0.0212 and the rank correlation between similarity advantage and accuracy advantage from 0.762 to 0.755. Excluding this person as target, donor and penalty-validation subject gave 0.0218 and 0.748. The original penalty was reselected in every remaining comparison. The full cohort remains the main analysis. [Exclusion audit](flagged_sensitivity.json)

The pooled-decoder experiment's prespecified result remains zero calibration examples saved at the tested R² threshold. At that threshold, the two augmented decoders' held-out predictions had nearly converged. [Calibration and prediction checks](../nki-split-half/README.md)
