# Scan density, wiring and recall gains

At eight target examples, the raw similarity association in NKI persisted after adjusting for the relative mismatch in scan density. The target-level Spearman correlation changed from 0.720 to a partial rank correlation of 0.680 (95% bootstrap interval 0.318 to 0.904). In Beijing it changed from 0.151 to 0.155 (−0.149 to 0.435). Both estimates use equal source contributions from each donor.

## Scan counts and repeatability

There are 3,403 possible undirected interregional edges in each 83-region matrix. We counted positive weights above the matrix diagonal. Across both visits, NKI had a median of 418.5 edges (range 177–626) and Beijing a median of 1,567.5 (1,291–2,155). Repeat-scan absolute-agreement single-measure ICC(A,1) was 0.437 (0.052 to 0.698) in NKI and 0.589 (0.292 to 0.778) in Beijing. The median within-person absolute change was 77 edges in both cohorts. Interval estimates for ICC resample whole participants. [Every scan](scan_counts.csv) · [Distributions](count_distributions.csv) · [Repeatability](test_retest.csv)

## Matched-pair analysis

Each directed comparison reports the change in mean held-out R² from replacing one donor first scan with the target's first scan in the common bank. Pair-level estimates average six calibration draws and ten delays. The first predictor is the own-minus-donor difference in raw similarity to the target's second scan. The density predictor is the absolute donor-to-target density gap minus the absolute own-to-target gap. Positive differences favor the personal scan. Density is an edge fraction between 0 and 1. All metrics are within cohort.

Separate Gaussian models use crossed independent random intercepts for target identity and comparison donor. We fit fixed slopes and variance components by restricted maximum likelihood (REML), with a residual variance and two nonnegative variance ratios. We report approximate model-based 95% intervals using the GLS covariance and a t reference with number of target participants minus one degrees of freedom. We also fit marginal ordinary least squares regressions and independently resample target and donor identities 4,000 times, weighting each observed pair by the product of its target and donor multiplicities. The latter intervals account for repeated roles among observed assignments and can differ from the crossed-model intervals. The analysis keeps training fits, streams and sampled target–donor comparisons fixed. The two random-intercept roles can share participant identities without imposing equality between their effects.

| Eight examples, equal donor contributions | NKI (75 pairs) | Beijing (171 pairs) |
|---|---:|---:|
| Unadjusted change in R² per 0.10-unit similarity advantage | +0.00687 [0.00431, 0.00943] | +0.00123 [−0.00040, 0.00286] |
| Density-adjusted change per 0.10-unit similarity advantage | +0.00676 [0.00405, 0.00947] | +0.00106 [−0.00061, 0.00274] |
| Marginal adjusted slope per 0.10, with target/donor bootstrap | +0.00754 [0.00148, 0.01408] | +0.00099 [−0.00242, 0.00395] |

Giving the added first scan 25% of all source examples yielded a mixed-model density-adjusted slope per 0.10 similarity of +0.0170 [0.0109, 0.0230] in NKI and +0.0076 [0.0016, 0.0135] in Beijing. Beijing's interval in the marginal crossed-bootstrap analysis included zero. The equal-donor fraction is smaller in Beijing (1/56) than NKI (1/24). [All models and variance components](crossed_models.csv) · [Target-level partial ranks](target_partial_rank.csv)

We also split similarity into Jaccard overlap of positive edges (intersection divided by union) and Pearson correlation of weights restricted to edges present in both scans. The smallest shared-positive set contained 35 edges in NKI and 803 in Beijing. Across the assigned pairs the raw similarity advantage correlated with both the Jaccard advantage and the shared-weight advantage (Spearman 0.869 and 0.963 in NKI, 0.407 and 0.999 in Beijing). This indicates that the original metric retained substantial information about edge weights. In a joint NKI model with density, a 0.10 Jaccard advantage predicted +0.0127 R² [0.0058, 0.0196], while a 0.10 shared-edge weight-correlation advantage predicted +0.0001 [−0.0032, 0.0034]. The marginal Jaccard slope's crossed-bootstrap interval included zero. The split describes an association with edge overlap. A causal interpretation needs separate evidence. Some Beijing models fitted the donor-intercept variance on its zero boundary. All fits converged. [Predictor correlations](predictor_correlations.csv) · [Pair features and results at every original budget](matched_pairs_by_budget.csv)

## Cross-cohort slopes and overlap

The NKI-minus-Beijing density-adjusted slope difference per 0.10 raw similarity advantage was +0.00570 [0.00249, 0.00892] in the crossed-intercept model. In the marginal regression with crossed resampling it was +0.00656 [−0.00035, 0.01389]. Transferability is uncertain given the sensitivity to model assumptions, cohort differences and the shared donor banks. Restricting NKI to Beijing's full observed person-level similarity-advantage range left ten participants, with Spearman ρ = 0.53. Restricting to Beijing's central 90% left five and gave ρ = 0.00. Those slices illustrate range sensitivity and support size, with limited precision. [Slope comparisons](slope_difference.csv) · [Range-restricted results](restricted_range.csv)

## Results across label budgets

The primary equal-donor associations at 0, 8, 16 and 32 labels were 0.718, 0.720, 0.423 and 0.173 in NKI and 0.208, 0.151, 0.194 and −0.039 in Beijing. At 64 or more labels, the largest absolute target-average R² gain for either cohort with equal donor contributions was 0.000127. Correlations of these very small differences are numerically defined but have limited predictive relevance. The full table includes all six prespecified budgets plus zero, paired gains, source-arm scores and participant intervals.

The early area under the performance curve uses trapezoidal integration over 0, 8, 16 and 32 labels, divided by 32 to stay in R² units. In NKI the equal-donor personal gain was 0.00367 [−0.00115, 0.01078], and the 25% gain was 0.00933 [−0.00081, 0.02159]. In Beijing those gains were 0.00051 [−0.00117, 0.00189] and 0.00539 [0.00167, 0.00920]. Across the six equally weighted budgets (8 to 256), the previously reported 25% gains were 0.00397 in NKI and 0.00232 in Beijing. These differ from the early area and from the zero-label comparison. [Every budget and normalized early area](budget_curves.csv)

The runner matches the previous eight-label pairwise gains and raw correlations for every comparison. A separate statsmodels implementation with a different optimizer reproduced all primary raw and density-adjusted fixed slopes in both cohorts within 0.0000014, and a separate residual-rank calculation corroborated the partial correlations. [Independent numerical checks](independent_check.json)

```sh
python scripts/density_associations.py
python scripts/verify_density_models.py
```
