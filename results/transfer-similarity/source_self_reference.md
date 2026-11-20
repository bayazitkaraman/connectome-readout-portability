# How much accuracy does the frozen decoder lose when only its wiring input changes?

We fitted each personal readout on 2,352 labeled source-graph time steps. For each compared identity pair we retained the original transfer penalty chosen using the other 23 NKI or 55 Beijing identities. We then used the **identical readout weights and intercept** on both the source and destination reservoirs. Both reservoirs received the same four fresh input sequences (2,048 scored time steps). This gives a paired change attributable to substituting the estimated wiring in the simulation, conditional on the trained readout and fixed streams. Acquisition, reconstruction effects and possible biological change are combined within the between-visit substitution.

| Cross-visit contrast | Source scored with source-tuned readout | Source scored with transfer-tuned readout | Repeat scored with that frozen transfer-tuned readout | Source minus repeat R² (95% paired interval) |
|---|---:|---:|---:|---:|
| NKI, odd first to odd second | 0.9651 | 0.1910 | 0.0428 | 0.1481 [0.0992, 0.2050] |
| Beijing, first to second visit | 0.8309 | 0.2373 | 0.1526 | 0.0846 [0.0657, 0.1061] |

The first column reports local tuning on an additional 1,176 labeled source-graph validation steps. The next two columns use the same transfer-selected readout. Its source accuracy was already lower before the graph substitution by 0.7741 R² in NKI and 0.5936 R² in Beijing. The change in regularization therefore accounts for a large part of the numerical distance between high within-graph accuracy and zero-label transfer. When the extremely weak locally selected relative penalty of 10⁻¹⁴ was instead frozen for transfer, prediction errors were very large in both primary comparisons. This behavior illustrates how accurately a readout can fit a particular wiring estimate while generalizing poorly to another estimate. [All archived-grid means and paired intervals](source_self_summary.csv)

## Holding regularization fixed in NKI

At a common relative ridge penalty of 10, source and destination were stimulated with identical test inputs in each row. Within each target visit the target signal and even-direction graph are shared between the two comparisons. The source-graph scores can differ slightly between rows using the same graph because its test inputs are determined by the destination visit.

| Odd-direction source | Even-direction target | Source R² | Destination R² | Paired loss [95% interval] |
|---|---|---:|---:|---:|
| Visit 1 | Visit 1 | 0.1896 | 0.1526 | 0.0370 [0.0199, 0.0550] |
| Visit 2 | Visit 1 | 0.1881 | 0.0586 | 0.1295 [0.1025, 0.1565] |
| Visit 2 | Visit 2 | 0.1899 | 0.1763 | 0.0136 [0.0041, 0.0251] |
| Visit 1 | Visit 2 | 0.1910 | 0.0475 | 0.1435 [0.0973, 0.1930] |

The common-penalty and planned cross-visit-selected-penalty results agreed for all four odd-to-even conditions. [Fixed penalty results](source_self_fixed10_summary.csv) · [Cross-visit-selected penalty results](source_self_cross_tuned_summary.csv)

The finer ridge grid also produced a positive paired source-to-repeat loss. NKI's mean changed from 0.1592 to 0.0411, a loss of 0.1181 [0.0772, 0.1669]. Beijing's mean changed from 0.2248 to 0.1550, a loss of 0.0698 [0.0549, 0.0863]. All intervals use 10,000 paired resamples of targets, conditioning on fitted readouts, regularization choices, complete comparison bank, and simulated streams. The zero-label comparison refers only to labels used to fit or select the transferred readout for the target. Its scored outcome always uses known simulated labels. [Fine-grid paired estimates](source_self_dense_summary.csv)

Run from the project root after the existing single-donor, split-half and refined-grid analyses.

```sh
python scripts/source_self_reference.py
python scripts/source_self_reference.py --grid fixed10
python scripts/source_self_reference.py --grid cross_tuned
python scripts/source_self_reference.py --grid dense
```
