# V35P distribution results

## Single-release fidelity

| Method | Covariance discrepancy | Energy distance | MMD | Fiber variance retained |
|---|---:|---:|---:|---:|
| Fiber-OneStep | 0.941168 | 4.315038 | 0.156918 | 0.179976 |
| Fiber-Stratified-Resample | 1.027775 | 1.169124 | 0.035192 | 1.035694 |
| Fiber-SANDiff | 0.913187 | 1.699086 | 0.061687 | 0.784008 |

Fiber-SANDiff retains the best covariance discrepancy, but simple stratified
resampling is better on energy distance, MMD, and variance calibration. Retrained-head
balanced accuracy is also slightly higher for resampling (`0.352623` versus
`0.350116`); the fixed-head output is exactly identical for both.

## Sixteen-release diagnostics

| Method | Within-H variance | Duplicate rate | Diversity | Nearest training-fiber distance |
|---|---:|---:|---:|---:|
| Fiber-OneStep | 0.0000 | 1.0000 | 0.0000 | 5.5433 |
| Fiber-Stratified-Resample | 72.5354 | 0.00637 | 11.4554 | approximately 0 |
| Fiber-SANDiff | 52.9137 | 0.0000 | 9.3069 | 6.5064 |

OneStep is the registered zero-diversity control. Resampling has occasional repeated
donors but greater empirical diversity; SANDiff produces unique outputs but remains
under-dispersed relative to the empirical population fiber channel. Near-zero donor
distance for resampling is structural and is not treated as an independent quality
win.

All 16 releases per query contributed to these summaries. No target-selected sample
or favorable-release selection was used.
