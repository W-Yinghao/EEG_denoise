# V35P terminal report

## Outcome

V35P corrected the privacy interpretation and completed the missing simple stochastic
baseline. All strong exact-fiber channels share the frozen H-visible privacy boundary.
Fiber-Stratified-Resample outperformed Fiber-SANDiff on energy distance, MMD, variance
calibration, and retrained utility, so the final positioning is:

```text
C. Fiber-Stratified-Resample is clearly preferable
```

The exact function-preserving population fiber method remains supported; unique
diffusion-specific empirical superiority is withdrawn.

## Git lineage

```text
base:          e10dd40100e60f5e47c4d1a917ec4515880fc9ca
implementation:4905872219c24b14494eb91a06f32ad6d3f28ff6
attack:        e15727f2bac559fa36fda0ad0b4ba85aa28e8cde
distribution:  7879fe14dab2343da96f583e2169ed459b0189f3
ledger v3.4:   20cb7d7ef42da047f216e9d22102c32d3701369b
terminal:      recorded after this report is committed
```

## Execution and governance

```text
Slurm:                 940652_[0-5]
accepted:              6/6
failed/recovery:       0/0
checkpoint SHA:        18/18
targeted tests:        38/38
clean archive tests:   38/38
latency benchmark:     not run
waveform sealed reads: 0
V33P/V34P:             unchanged
A-track:               unchanged at 0c4f2301c1f873120fe54537cde3c76fff7ea3a2
manuscript:            unchanged and not compiled
```

No PR was created and master was not merged. The unrelated job `936612` was not
touched; no V35P jobs remain running.
