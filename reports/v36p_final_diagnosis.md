# V36P final diagnosis

## Outcome

V36P completed 6 folds × 2 seeds on all 54 OpenBMI participants. Every participant appeared once
in outer test; seeds were repeated measurements, not biological samples. All 12 primary jobs and
12 frozen-checkpoint task-utility recovery jobs completed with empty stderr. Forty-eight checkpoint
bindings passed SHA256 verification. Waveform sealed reads were zero; A-track and manuscript were
unchanged, and no latency benchmark was run.

External exact-fiber replication succeeded: fixed-head BA was `0.734537`, exact channels caused no
prediction or BA change, and registered strong-channel adaptive leakage equaled the H-only boundary
(`0.151667`) rather than RAW A_Z leakage (`0.349815`).

The deployment distinction also replicated. Resampling directly released training fibers (exact
copy `1.0`, exposure score `0.998454`); Gaussian and SANDiff had no exact/near copies, with scores
`0.000090` and `0.002392` respectively.

The diffusion-specific comparison is negative. Fiber-SANDiff was worse than the simple model-only
Gaussian on energy distance (`0.8809` vs `0.5080`), MMD (`0.03033` vs `0.01424`), covariance
discrepancy (`0.6882` vs `0.6566`), and variance calibration (`0.8688` vs `1.0077`). Energy, MMD,
and variance favored Gaussian in 12/12 cells. Retrained task utility was practically similar.

## Final positioning

```text
B. Model-only stochastic channels equivalent at the exact-channel/privacy boundary;
   Gaussian empirically preferable for this cohort.
```

This is a narrowed use of category B: Gaussian and SANDiff share exact task preservation,
H-bounded source privacy, and exemplar-free deployment, but they are not empirically tied on
distribution fidelity. The registered Gaussian is the simpler and stronger model-only channel.
The evidence does not support a diffusion-specific superiority claim, while category C is also
incorrect because bank resampling is not the only strong distribution method.

The supported conclusion is therefore: exact function-preserving, model-only population fiber
channels replicate externally and avoid bank-based exemplar release; current Fiber-SANDiff is a
valid implementation but does not outperform the fixed conditional Gaussian baseline. No formal
anonymity, exact CMI, universal encoder/task generalization, or clinical claim is made.
