# V34P terminal report

## Outcome

V34P completed the exact-head-fiber method on all three outer folds and both seeds.
The selected positioning is `A. Fiber-SANDiff positive method`: fixed decisions are
preserved exactly, participant-first privacy improves over RAW, and stochastic
replacement better preserves the pooled fiber distribution than the conditional-mean
control. Fiber-OneStep remains stronger in retrained utility and much lower in cost.

## Git lineage

```text
base:           5292c1a552ca3fd5980f37291cd53a98ab6d01ea
implementation: 9d8b34b201a1e22d677aec9c4641c609ae33e57d
geometry:       fb9eab31374398c7ec7fcceff4b187135679354d
selection:      48baa16b2b43840cc38f9c001093f55987ca0171
result:         21a3e7fbb2e354d186c386f795275ac87af8618e
ledger v3.2:    664fc754524e731424e78934b14427b29d056199
terminal:       recorded after this report is committed
```

## Execution and verification

```text
Slurm:                 940575_[0-5]
accepted:              6/6
failed/recovery:       0/0
checkpoint SHA:        24/24
targeted tests:        24/24
clean archive tests:   24/24
waveform sealed reads: 0
manuscript:            unchanged, not compiled
V25–V33P:              unchanged
A-track:               unchanged at 0c4f2301c1f873120fe54537cde3c76fff7ea3a2
```

The only current Slurm job is the unrelated pre-existing `936612`; no V34P jobs
remain. No PR was created and master was not merged.
