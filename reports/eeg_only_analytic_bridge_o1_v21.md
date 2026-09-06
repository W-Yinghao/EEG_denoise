# O1-V21 — EEG-Only Analytic Artifact-Bridge Gate

Scientific route: `O1_EEG_ONLY_AMPLITUDE_NOT_IDENTIFIED`
Terminal label: `SPATIAL_TRANSFER_VALID_EEG_ONLY_TEMPORAL_COEFFICIENT_NOT_IDENTIFIED`
DET authorization: `NOT_AUTHORIZED`.

## Primary participant-first results

- U_P=-0.059566, relative=-5.630%, unrestricted p=0.97397, sign-flip p=0.996094, positive=1/15, bootstrap=[-0.16852509063926066, -0.0023424902074358285].
- U_W=-0.094335, relative=-9.219%, unrestricted p=0.97397, sign-flip p=0.996094, positive=1/15, bootstrap=[-0.2690667087993841, -0.002764784762921435].

## Detector and safety

- Detector AUROC=0.857252; TAR@FAR5=0.063060.
- Preservation=0.988844; PSD distortion=0.005636; covariance distortion=0.016283.
- Outside-mask max=0; off-span max=1.15e-15.
- ERP U_P=-0.016826; SSVEP U_P=-0.102306.

## Oracle bottleneck decomposition

- D1 oracle-mask U_P=-0.054648, U_W=-0.090487.
- D0 primary risk=1.117637; D2 query-operator risk=1.213980; D3 oracle-mask+query-operator risk=1.130786; D4 max risk=0.

O1 uses query EEG only at inference. Query EOG and Qgen operator were opened only after output freeze by evaluator stages. No model, GPU, raw signal, sealed outcome, manuscript, DET, or diffusion operation ran.

## Decision boundary

The primary analytic bridge did not convert V20 operator-transfer evidence into deployable EEG-only action. Both participant-level owner effects were negative, the detector missed the frozen TAR@FAR=5% requirement, and neither oracle timing nor query-operator diagnostics isolated a support-operator or mask-only repair. Exact query-EOG subtraction remained a valid nondeployable ceiling. Therefore the frozen route is `O1_EEG_ONLY_AMPLITUDE_NOT_IDENTIFIED`; DET and diffusion are not authorized.

Absolute waveform safety passed (outside-mask identity exactly zero; off-span ratio `1.15e-15`; preservation `0.988844`; PSD distortion `0.005636`; covariance distortion `0.016283`). Population-relative safety did not pass because SSVEP utility and the preservation/PSD/covariance utilities were adverse despite remaining within the absolute bounds.

## Engineering recovery lineage

P6 job 934575 incorrectly used detector-empty units for the context-intervention fixture and was superseded by active-mask recovery 934579. The first P7–P12 chain serialized correction arrays as float32, producing a serialization-only off-span residual of `2.41e-8`; that chain is retained as superseded. The isolated float64 recovery (P7 934649_[0-14], P8 934665, P9 934668_[0-14], P10 934684, P12 934686) restored the off-span residual to `1.15e-15` without changing any scientific setting. Primary effects changed only at approximately `1e-11` scale.
