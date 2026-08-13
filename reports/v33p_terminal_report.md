# V33P terminal report

## Git lineage

```text
base:           2b1522e79a5b701389b1446f51589a9862fb5f15
implementation: b0074413d69a088956fa0e02d42415c8d5928f64
selection:      afb49f10c6e0ca266152553752900938467eec9e
refit/result:   887de1e4deac2bb79a4c95d67f470f83105fb6ca
ledger v3.0:    382c9d1695da9ffa429b2073e52b51ef3307be1b
terminal:       commit containing this report and manifest
```

Remote/local parity is checked after the terminal commit. No PR is created and
master is not merged.

## Execution and validation

```text
accepted jobs: 940211_[0-5]
failed jobs: none
recovery jobs: none
current V33P jobs: none
checkpoint SHA: 30/30 verified
targeted tests: 29/29
clean archive tests: 29/29
```

## Scientific outcome

Strong SANDiff relative to RAW reduced adaptive subject accuracy by 0.079861
and participant-averaged verification AUROC by 0.053964, with 9/9 participants
positive for both privacy utilities. Fixed-head BA changed by −0.005787. The
full-sampler checkpoint improved fixed BA by 0.000579 and adaptive privacy by
0.001350 over the single-timestep checkpoint, but both effects were
heterogeneous.

Final positioning:

```text
SANDiff and one-step practically equivalent
```

## Governance

```text
waveform sealed reads: 0
query EOG reads: 0
V32P results: unchanged
A-track: unchanged at 0c4f2301c1f873120fe54537cde3c76fff7ea3a2
manuscript: unchanged and not compiled
privacy-weight repair: not used
```
