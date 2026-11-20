# Lausanne frozen-readout feasibility protocol
Recorded before scoring the real networks. This pilot measures portability on simulated signals. It does not estimate human memory ability or the anatomical advantage over matched synthetic cohorts.

## Data and model

Use all 70 individual structural matrices from the Lausanne release, at the coarsest supplied matrix resolution (68 cortical indices). Preserve the released order. Rescale each subject's weighted, undirected matrix to spectral radius 0.90. Use the published conn2res tanh EchoStateNetwork, zero initial state and no leak mixing. Compare 0.70 and 1.00 as secondary regimes. The input sequence is independent uniform noise on [-1, 1]. Eight deterministic, atlas-aligned input nodes receive the same scalar with four weights +0.5 and four -0.5. Read out all other nodes. This is a generic input placement with no regional or modality-specific claim. Each sequence starts independently, with 200 simulated steps discarded.

## Splitting and fitting

The native matrices contain large regional diagonal entries. Set every diagonal to zero to model interregional recurrence, then rescale. Keep all 68 atlas columns, including any isolated region, so readout weights always refer to the same node.

- Fixed fivefold cross-validation by participant, random seed 20261119. Each of 70 participants is tested once with the shared decoder fitted on the other 56. Test on four independent 512-step streams per person, disjoint from all training and validation streams. Two time points from different simulated streams are never used to form a delayed target.
- Fit a linear ridge readout with an unpenalized intercept using the current states to predict inputs 1 through 10 steps ago. Select one relative ridge penalty across delays from 1e-12, 1e-10, 1e-8, 1e-6, 1e-4 and 1e-2, scaled to the mean training feature variance; choose by minimum mean validation error. Store selected penalties.
- Equal primary fitting budgets: the shared decoder uses 42 evenly spaced samples from each of 56 participants (2,352 total); the individual decoder uses 2,352 samples from its own network. For shared penalty selection, separate 42 inner fitting people (56 samples each) from 14 inner validation people (84 samples each, 1,176 total). For individual tuning use a separate 1,176-step validation sequence. At final scoring, no test person's wiring, states, scores, or labels enter the pooled fit or its tuning. The individual reference is explicitly allowed its own calibration sequences.
- At primary gain, compare with a decoder fitted on the mean raw structural matrix of the 56 training people (same sample/validation budgets). Also test additional pooled training data (all 2,352 fitting rows per training person); label this as an unequal-budget sensitivity analysis. For these and the primary readout use the same held-out participant streams.

## Outcomes

Primary: mean across participants and delays of (shared MSE minus individual MSE) divided by the known input variance 1/3. Report each model's ordinary held-out R² as well, plus the paired loss in R² points. Report delay 9 explicitly, and all 10 delays, so a floor at long delays is visible. Signed losses are retained. Positive loss means individual calibration helps. Also report gains 0.70 and 1.00. Bootstrap the 70 participant-level paired summaries (10,000 draws) for a descriptive 95% interval conditional on this fitted cross-validation run. It does not incorporate cohort, input-map or training-set resampling. No synthetic wiring control, reliability analysis or behavioral association is part of this feasibility estimate.

