# Zero-label transfer from a single first scan

At zero target calibration examples, Beijing's participants scored mean held-out R² of 0.15265 with their own first-scan decoder and 0.10346 with an average other person's individually fitted decoder. The gain was +0.04919 (participant-bootstrap 95% interval 0.02502 to 0.06960). NKI scored 0.04284 versus 0.02376, for a gain of +0.01908 (−0.02719 to 0.06118). Personal decoders beat their mean comparison decoder in 48 of 57 Beijing targets and 15 of 25 NKI targets.

We evaluated every ordered first-scan source choice on each participant's second-scan recall signals, comprising 600 own-versus-other comparisons in NKI and 3,192 in Beijing. For any given pair, transfer ridge strength was selected by validating baseline fits on separate second-scan signals for the remaining 23 or 55 identities. Both candidate decoders used that same relative ridge penalty. The comparison averages separately scored donor decoders. The target received zero calibration labels, with 2,048 held-out simulated time steps scored at ten delays (stream role 61). This is the test stream used in the previous few-label calibration analysis. The original NKI frozen-transfer report used a different test stream (role 2).

Repeating the analysis with an identical 2,352-step source training signal for every first scan gave gains of +0.01867 in NKI and +0.04901 in Beijing. The primary run retained the original per-source training signals. All comparisons for a target use the same held-out test set. The intervals resample the target participants 10,000 times and condition on the complete donor bank. A replay of the original personal-only fits with their original leave-one-target-out penalties verified the zero-label scores against the previous reports. [Zero-label results](summary.csv) · [Participant averages](single_donor_participant_means.csv) · [All individually scored donors](single_donor_pairs.csv) · [Replay check](verification.json)

```sh
python scripts/frozen_single_donors.py
```
