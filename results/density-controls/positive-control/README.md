# Can the pooled-scan comparison detect personal information?

We planted a stable code in six of ten memory channels of a simulated linear delay line. The decoders received the same number of source samples and exactly the same target examples within each comparison. Turning off the injected code gave zero own-minus-other R² for every comparison and budget. At unit injection, pairs sharing the same assigned sign also gave exactly zero, which verifies that the paired source inputs match. [Plan and analytical reference](PROTOCOL.md)

| Unit injection | Own-minus-other R² at zero target labels | Own-minus-other R² at eight labels | First tested budget sustaining R² ≥ 0.50, own / other |
|---|---:|---:|---:|
| Equal contribution per donor | +0.0554 | +0.0232 | 8 / 8 |
| Added scan supplies 25% | +0.3305 | +0.1449 | 0 / 8 |

With equal shares, the two decoders tied at the recall threshold even though early accuracy detected a known personal effect. A larger share exposed an eight-example saving. The observed oracle R² was 0.9408, close to the analytical value of 0.9412. This synthetic fixture checks detection of a large, stable signal and the resolution of the threshold in this design. Estimating power for the much smaller observed human-network effects requires a separate, calibrated model of scan-to-scan variability. [Scores across all budgets](curves.csv) · [Summary](summary.json)

From the project root run `python scripts/positive_control_scan_swap.py`.
