# V36P head-aware source privacy

The EEGNet task function was not clearly underfit: participant-first outer-test fixed-head balanced
accuracy was `0.734537`; Stage-A validation balanced accuracy ranged from `0.708` to `0.764` in the
first six audited cells and selected epochs ranged from 20 to 60 across all cells.

For the adaptive MLP, H-only balanced accuracy was `0.151667`. Primary finite-threat leakage
`max(A_H,A_Z,A_HZ)` was also `0.151667` for Fiber-OneStep, Fiber-Gaussian,
Fiber-Stratified-Resample, and Fiber-SANDiff. Thus all strong channels reached the registered
H-visible boundary. The corresponding RAW A_Z attack was `0.349815`.

Participant-first RAW-minus-A_Z adaptive recall reductions were:

| Method | Mean | 95% participant bootstrap CI | Positive participants |
|---|---:|---:|---:|
| Fiber-OneStep | 0.214630 | [0.169072, 0.261574] | 49/54 |
| Fiber-Gaussian | 0.224815 | [0.183889, 0.268150] | 51/54 |
| Fiber-Stratified-Resample | 0.225000 | [0.187315, 0.263799] | 51/54 |
| Fiber-SANDiff | 0.227685 | [0.184630, 0.269722] | 52/54 |

Cross-session same/different AUROC was `0.53846` for RAW, versus `0.50115` Gaussian, `0.50392`
Resample, and `0.50500` SANDiff. `CE(A_H)-CE(A_HU)` was non-positive for every strong method and
both registered attacker families on average. This is a finite-attacker closure diagnostic, not a
CMI estimator. No strong channel is credited with privacy below A_H.
