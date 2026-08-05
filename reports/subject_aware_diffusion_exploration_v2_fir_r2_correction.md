# R2 FIR scientific correction

This document is an additive correction to the immutable results produced at
commit `8c54fc1`. It does not overwrite the historical CSV, JSON, figures, job
records, or coverage counts.

`scientific_status = invalidated_fir_implementation_and_absolute_performance`

The completed-job coverage remains 16/16 Klados source records and 58/59
SGEYESUB participant stems. All R2 scientific contrasts, rankings, confidence
intervals, and uncertainty comparisons from that execution are invalid and
require a repaired rerun. They are evidence for neither a diffusion failure
nor a subject-awareness failure.

## Absolute-performance audit

On Klados for the structured R2 route, mean clean-waveform RRMSE was RAW
`1.2948`, POP `0.3990`, DET-MATCH `6.9105`, and DIFF-MATCH-K8 `5.8163`.
The positive DIFF-minus-DET utility arose because the deterministic arm was
even worse; the diffusion output itself was substantially worse than both RAW
and POP. The earlier phrase “FIR carrier 下存在 diffusion-vs-deterministic
机制信号” is withdrawn.

On SGEYESUB, structured DIFF-MATCH had EOG reduction `0.2453`, non-artifact
preservation `0.6883`, and covariance distortion `0.1886`; POP was better on
all three endpoints at `0.3231`, `0.7762`, and `0.0938`, respectively. R2 was
therefore absolutely dominated by POP under the executed implementation.

## Implementation defect and repair contract

`_fir_design` is lag-major. The historical code directly reshaped flat ridge
coefficients as `(C,E,L)`, but the correct conversion is
`reshape(C,L,E).transpose(0,2,1)`. The repair adds a two-EOG, five-lag
fit→cache→runtime impulse round trip and explicit inverse conversion.

The repaired audit uses blocked A→B/B→A support cross-fitting for FIR and
state-specific operators. FIR shrinkage/reliability is selected from FIR's own
held-out support prediction error, with lag locations specified in
milliseconds and converted independently for each sampling-rate cell.

Repaired R2 uses only:

`H_rho = H0 + rho * (Hs - H0)`

`x_hat = y - H_rho * a_hat`

Deterministic and diffusion estimates share the same canonical artifact
target and the same reconstruction. DET-POP and DIFF-POP must run their actual
population-conditioned estimators. `mechanism_g1` sets both the model
condition and reconstruction reliability to one.

Route ranking is prohibited until output scale is finite, Klados beats RAW
with positive delta-SNR, and SGE is not jointly dominated by POP on artifact
reduction, preservation, and covariance distortion. Equal-attenuation
comparison excludes the gamma-zero population identity endpoint and uses only
the mutually reachable nonzero attenuation interval.

The independent full-C/FiLM/activity-gate/adapter/guidance/SDEdit GPU work on
`codex/parallel-subject-explore` does not import the R2 implementation or its
cache and continues separately.
