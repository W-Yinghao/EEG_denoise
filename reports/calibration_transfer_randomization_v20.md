# V20 — Participant-Level Calibration-Transfer Randomization Gate

Scientific route: `V20_NATURAL_TRANSFER_PASS`

Terminal label: `NATURAL_CALIBRATION_TRANSFER_ESTABLISHED`

O1 authorization: `O1_AUTHORIZED_NOT_RUN`.

V20 is a new development-only protocol, not an exact recovery or retrospective validation of v19.

## Primary natural-query result

- N_P = +0.178803; relative improvement 20.267%; one-sided participant-label randomization p=9.9999e-06.
- N_W = +0.174162; relative improvement 19.845%; one-sided participant-label randomization p=9.9999e-06.
- N_P median=+0.149810, 15/15 positive, descriptive participant bootstrap CI [+0.110196, +0.269107].
- N_W median=+0.127289, 15/15 positive, descriptive participant bootstrap CI [+0.112001, +0.265159].
- Two-sided randomization sensitivity: P=9.9999e-06; W=9.9999e-06.
- P endpoint pass: True; W endpoint pass: True. Both are required by an intersection–union gate.
- Scientific n=15 primary recipients; policy denominator=16 with sub-24 fallback zero reported only descriptively.

The endpoint is natural query operator prediction risk, not clean-EEG reconstruction error. Query EOG is evaluator-only.

## Construct controls

- TIME_SHIFT passed: True.
- CHANNEL_PERM passed: True.
- TIME_SHIFT minus MATCH mean=+0.275205, sign-flip p=6.10352e-05.
- CHANNEL_PERM minus MATCH mean=+0.238251, sign-flip p=3.05176e-05.
- Gain-normalized direction preserved: True (N_P=+0.102476; N_W=+0.104140).

## Task and paired sensitivities

- ERP: N_P=+0.084226; N_W=+0.158487.
- SSVEP: N_P=+0.273379; N_W=+0.189837.
- Historical O0-B: H_P=+0.768602; H_W=+0.706231; oracle max error=1.31e-15.
The historical paired O0-B result remains only a controlled paired mechanism signal and cannot rescue the natural gate.

## Reproducibility and boundaries

- 100,000 accepted fixed-point-free injections; PCG64DXSM seed 20260820; manifest `c6b3dcc3383f980cf05c43ca29f326ee1ddef37048244af1ec3d1f2c7760adcc`.
- Independent long-form and dense/vectorized implementations agreed to 0 (tolerance 1e-12).
- No positive clipping, q95 floor, participant-specific floor, time-shift null replicate, or row-level pseudo-replication was used.
- No raw or sealed signal was opened. No GPU, O1, DET, diffusion, manuscript, or confirmation operation ran.
