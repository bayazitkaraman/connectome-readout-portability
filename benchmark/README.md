# Apply the benchmark to another repeat-scan dataset

Start with one structural connectivity matrix per participant per visit. The benchmark generates the input signals and recall labels. Run the commands below from the repository root using Python 3.10.19.

## Run your own repeat-scan data

**1. Export the matrices.** Use the same atlas, parcel definitions, and edge-weight convention at every visit. The importer reads square CSV matrices with parcel IDs in the first row and first column. It aligns rows and columns by those IDs, so different file orderings are fine. Each scan must have the same parcel set and finite, nonnegative, symmetric weights. The simulation drops the diagonal. Here is a three-parcel example of the file layout:

```csv
parcel,Left_A,Right_A,Left_B
Left_A,0,4,1
Right_A,4,0,2
Left_B,1,2,0
```

**2. List the scans.** Put a file named `scans.csv` beside your matrices. Use the same participant ID at every visit and give each participant exactly one matrix per visit. Paths in this example are relative to `scans.csv`; subject IDs such as `001` are kept as text.

```csv
subject,session,path
001,baseline,matrices/001_baseline.csv
001,followup,matrices/001_followup.csv
002,baseline,matrices/002_baseline.csv
002,followup,matrices/002_followup.csv
003,baseline,matrices/003_baseline.csv
003,followup,matrices/003_followup.csv
```

**3. Convert and run.** In these commands, the `my-cohort` directory containing `scans.csv` is beside the repository. Choose a new NPZ filename and a new output directory. The importer checks participant–visit pairs and parcel labels before it writes the archive.

```sh
python -m pip install -r requirements-benchmark.txt
python -m benchmark.prepare_csv ../my-cohort/scans.csv ../my-cohort/connectomes.npz
python -m benchmark ../my-cohort/connectomes.npz ../my-cohort/results --state-comparisons
```

The default configuration drives eight parcels and requires at least nine. For the three-parcel layout above, append `--input-count 2` to the **last** command. With two participants, you can run the same-person state comparison. Transfer-tuned other-person comparisons require at least three. If you already have a correctly formatted NPZ, begin with the last command. [NPZ array layout](#input)

**4. Read the result.** Open `results/state_summary.csv` inside your cohort directory and select the row with `source_visit=baseline` and `target_visit=followup`. The program also reports the reverse direction. In the selected row:

| Column | What it tells you |
| --- | --- |
| `source_r2` | How accurately the frozen decoder recalled the simulated input on its training graph, using fresh test signals. |
| `repeat_r2` | How accurately those same weights recalled the same input on the follow-up graph. |
| `isotropic_r2` | Expected recall on the first graph under noise matched to the size of the change in states. |
| `noise_minus_repeat` | Expected noise R² minus measured repeat R². A positive value means the noise control overestimated portability. The paired participant-bootstrap bounds are `noise_minus_repeat_lo` and `noise_minus_repeat_hi`. |

Open `state_pairs.csv` for each participant’s scores and chosen penalty. For example, in the five-person synthetic run below, expected recall under noise was 0.3083 and measured repeat recall was 0.2908 on the first-to-repeat comparison. The overestimate was **0.0175 R²**, with a 95% interval of 0.0044 to 0.0287. Your cohort will have its own values. A standard run evaluates one input map; the paper's headline averages ten maps within each participant. [Paper estimates](../results/headline-robustness/map_portability.csv)

## Try the synthetic example

```shell
python -m pip install -r requirements-benchmark.txt
python -m benchmark.example demo.npz
python -m benchmark demo.npz demo-results --penalties 0.01 1 10 --state-comparisons
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

To reproduce the main comparison for another cohort, run
`python -m benchmark my-study.npz my-results --state-comparisons`.
For each participant and visit direction, the readout is fitted on the source, and the
relative penalty is chosen on paired validation signals from the remaining participants.
The same fresh input drives both graphs. The tool reports source and repeat R², the
RMS state change as a percentage of source training fluctuation, the expected score under
independent isotropic noise of equal size, and the expected score under zero-mean noise
whose spatial covariance equals the displacement’s uncentered second moment. The latter
two are analytical expectations conditional on the observed scan pair. A separate alignment
uses `--alignment-steps` unlabeled target observations (default 2,352, stream role 219).
Each comparison uses all requested output lags, or the single product output. Supply one
value with `--penalties` to evaluate a fixed operating point.

| File | Interpretation |
| --- | --- |
| `matrix_seed*_visit*_to_*.npz` | Rows are source people, columns target people. `wiring` contains Pearson correlations of upper-triangle edge weights including zeros. `fixed_r2[i,j,k]` scores source i on target j at penalty k. `source_local_r2` and `transfer_tuned_r2` use their corresponding selection matrices. File names use zero-based visit indices. The file includes both actual visit labels and participant IDs. |
| `individual.csv` | Each fitted scan’s held-out R² by output and across outputs at every penalty. The selection flag identifies the locally tuned penalty. Same-scan results consume target training and validation labels. |
| `reliability.csv` | Spearman correlation and two-way absolute-agreement single-measure ICC(A,1) of **independently fitted scores** across participants between each pair of visits. The table reports each fixed penalty and the local selection rule. A constant score can make a metric undefined. |
| `targets.csv` | Own/stranger/closest-stranger transfer by target and selection rule, structural and decoder identification midranks, fractional top-one credit for ties, and source-local loss under a wiring substitution. Tied nearest strangers are averaged. |
| `summary.csv` | Participant means of the main comparisons, including identification rates. Participants are the units of averaging. |
| `similarity_curve.csv` | Descriptive mean R² and fractions reaching R² of 0.10 and 0.50 in each anatomical-similarity bin, separately for own and stranger. Includes counts of pairs and distinct targets. Intervals include their lower endpoint, and the last bin includes 1. |
| `run.json` | Input checksum, settings, random seeds, parcel maps, task targets and package versions. |
| `state_pairs.csv` | With `--state-comparisons`, matched-input displacement and R² comparisons for each own repeat, including the projected spatial loss and residual cross term. Their sum is the measured source-to-repeat loss. |
| `state_summary.csv` | Participant means for each seed and visit direction. Paired 95% percentile bootstrap intervals for noise minus repeat and aligned minus repeat use 10,000 participant draws at the selected penalties. |

For transfer tuning, the diagonal uses all remaining identities for tuning. The paired own-minus-other column in `targets.csv` gives the personal readout each stranger’s matched penalty before taking the difference. `source_local_r2`, `source_r2` and `frozen_wiring_drop` use the target’s held-out input on the source network, keeping stimulation identical when a fitted readout is tested on both wirings. A seed is an analysis setting. Pooling its output rows as additional participants would inflate precision.

Identification and nearest-donor selection treat differences within 10⁻¹² as tied. This absolute tolerance applies in Pearson correlation units for wiring and R² units for decoder identification. The study’s training-budget-matched pooled decoder appears in the study scripts and participant tables. The generic CLI evaluates single-source readouts. The pooled decoder and average of individually fitted stranger scores answer different questions.
