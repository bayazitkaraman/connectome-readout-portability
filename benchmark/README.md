# Apply the benchmark to another repeat-scan dataset

Run commands from the repository root. The tested interpreter is Python 3.10.19.

```shell
python -m pip install -r requirements-benchmark.txt
python -m benchmark.example demo.npz
python -m benchmark demo.npz demo-results --penalties 0.01 1 10
python -m benchmark demo.npz demo-product --task product --seeds 20261119 20261120
python -m unittest discover -s benchmark/tests -v
```

The synthetic fixture contains five fictional participants, two visits and sixteen parcels. Replace `demo.npz` with your own data to apply the same analysis. Outputs are written to a new directory after the run succeeds.

## Input

Use numeric NumPy arrays and integer or plain Unicode labels. The file is opened with `allow_pickle=False`.

| Key | Shape | Contents |
| --- | --- | --- |
| `weights` | `(visits, subjects, regions, regions)` | Symmetric, finite, nonnegative structural connectivity. Zero means no measured edge. Diagonals are removed. |
| `subjects` | `(subjects,)` | Unique participant identifiers, ordered identically across visits. |
| `sessions` | `(visits,)` | Unique acquisition labels. |
| `region_ids` | `(regions,)` | Optional aligned parcel identifiers. Defaults to zero-based indices. |

For example, `np.savez_compressed("my-study.npz", weights=W, subjects=ids, sessions=visits, region_ids=parcels)` saves a cohort. Every participant must have every visit, and each parcel index must refer to the same anatomical region throughout. The file format assumes the depositor has performed this alignment. At least two participants and two visits are needed. The tool also accepts a leading reconstruction axis, `(reconstructions, visits, subjects, regions, regions)`, when accompanied by a `halves` label vector. Select one with `--half odd`. Each run compares every ordered pair of distinct visits within that reconstruction.

## Tasks and validation

An independent scalar uniform signal drives eight parcels by default, four with input weight +0.5 and four with −0.5. The remaining parcels form the readout. Graphs are scaled to spectral radius 0.90 and simulated with the bundled conn2res tanh reservoir. On small parcellations set `--input-count` to an even number smaller than the number of parcels. The study seed is 20261119. Each additional seed draws a new input map as well as fresh input streams. To vary only the map with a fixed seed, change `--map-key`.

- `--task recall` predicts ten previous input samples. Each output has its own held-out R², calculated against that output’s test mean. The reported transfer score averages the ten R² values equally.
- `--task product` predicts the product of input samples at lags 1, 2 and 3. Use `--lags` to specify a different set on either task. Each test stream has its own warmup.
- Training uses 2,352 labeled steps per source. The source-validation stream has 1,176 fresh steps, and testing uses four further independent streams of 512 steps. All donors are scored against the same held-out input for a given target. `python -m benchmark --help` lists the controls.
- Every supplied fixed ridge penalty is evaluated. Each value multiplies the mean eigenvalue of that source’s training-state covariance. The intercept is unpenalized. The default grid comprises powers of ten from 10⁻¹⁴ through 10⁴. Local tuning picks the penalty minimizing source-validation MSE. Transfer tuning picks the penalty minimizing mean cross-visit validation MSE across other identities, excluding the target and the source. The latter rule is available with at least three participants. Both roles have zero fitting and tuning labels from the tested target in the frozen transfer comparison.

## Outputs

| File | Interpretation |
| --- | --- |
| `matrix_seed*_visit*_to_*.npz` | Rows are source people, columns target people. `wiring` contains Pearson correlations of upper-triangle edge weights including zeros. `fixed_r2[i,j,k]` scores source i on target j at penalty k. `source_local_r2` and `transfer_tuned_r2` use their corresponding selection matrices. File names use zero-based visit indices. The file includes both actual visit labels and participant IDs. |
| `individual.csv` | Each fitted scan’s held-out R² by output and across outputs at every penalty. The selection flag identifies the locally tuned penalty. Same-scan results consume target training and validation labels. |
| `reliability.csv` | Spearman correlation and two-way absolute-agreement single-measure ICC(A,1) of **independently fitted scores** across participants between each pair of visits. The table reports each fixed penalty and the local selection rule. A constant score can make a metric undefined. |
| `targets.csv` | Own/stranger/closest-stranger transfer by target and selection rule, structural and decoder identification midranks, fractional top-one credit for ties, and source-local loss under a wiring substitution. Tied nearest strangers are averaged. |
| `summary.csv` | Participant means of the main comparisons, including identification rates. Participants are the units of averaging. |
| `similarity_curve.csv` | Descriptive mean R² and fractions reaching R² of 0.10 and 0.50 in each anatomical-similarity bin, separately for own and stranger. Includes counts of pairs and distinct targets. Intervals include their lower endpoint, and the last bin includes 1. |
| `run.json` | Input checksum, settings, random seeds, parcel maps, task targets and package versions. |

For transfer tuning, the diagonal uses all remaining identities for tuning. The paired own-minus-other column in `targets.csv` gives the personal readout each stranger’s matched penalty before taking the difference. `source_local_r2`, `source_r2` and `frozen_wiring_drop` use the target’s held-out input on the source network, keeping stimulation identical when a fitted readout is tested on both wirings. A seed is an analysis setting. Pooling its output rows as additional participants would inflate precision.

Identification and nearest-donor selection treat differences within 10⁻¹² as tied. This absolute tolerance applies in Pearson correlation units for wiring and R² units for decoder identification. The study’s training-budget-matched pooled decoder appears in the study scripts and participant tables. The generic CLI evaluates single-source readouts. The pooled decoder and average of individually fitted stranger scores answer different questions.
