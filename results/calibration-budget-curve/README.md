# How much do additional target states and their labels help?

Each participant's first-scan decoder was scored on the repeat scan. The three strategies used the same held-out test streams. At each budget, the alignment strategy estimated parcelwise means and standard deviations on a prefix of a separate target stream. The supervised strategy also used the delay labels on that prefix to fit a ridge correction to the remaining errors. Other participants supplied the validation data for choosing the correction penalty. [Protocol](protocol.md) · [Scoring script](../../scripts/calibration_budget_curve.py)

| Mean held-out repeat R², at 64 target states | Frozen | Unlabeled alignment | Alignment and labeled correction | Labeled minus aligned, paired 95% interval |
|---|---:|---:|---:|---:|
| NKI, 25 people | 0.0428 | 0.0292 | 0.6998 | 0.6706 [0.6131, 0.7280] |
| Beijing, 57 people | 0.1526 | 0.1897 | 0.5226 | 0.3330 [0.3098, 0.3546] |

At eight states the unshrunk parcelwise alignment reduced mean accuracy in both cohorts. At 256 states its mean gain in Beijing was 0.0515 [0.0358, 0.0696], about 92% of the full-budget gain of 0.0560 [0.0405, 0.0739]. The full-budget NKI gain was 0.0085 [−0.0173, 0.0397]. When all 2,352 target labels were supplied, supervised correction reached 0.9524 in NKI and 0.8361 in Beijing. These scores describe learning on the target scan. The one nested calibration stream per participant and the external validation panel are fixed in these pointwise participant-bootstrap intervals.

The [cohort summaries](summary.csv) and [participant results](people.csv) contain all ten budgets, including zero. [Selected settings](penalty_selection.csv) and [donor validation losses](validation_losses.csv.gz) document the choice made for each held-out person. The [checks](checks.json) compare the frozen and full-budget estimates against archived scores and independently verify the residual model. [Figure 4](Figure_4.pdf) · [Plotting script](../../scripts/plot_calibration_budget.py)
