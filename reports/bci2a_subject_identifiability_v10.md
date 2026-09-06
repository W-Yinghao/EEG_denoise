# BCI2a subject identifiability V10

All nine BCI-IV-2a participants were audited. Query operators were evaluator-only and never entered deployment inference.

| protocol | MATCH-WRONG mean | positive | MATCH-POP mean |
|---|---:|---:|---:|
| same_session_T | +0.0968 | 5/9 | -0.0999 |
| same_session_E | +0.0894 | 5/9 | -0.0017 |
| cross_session | +0.0892 | 5/9 | -0.0902 |

BCI2a decision: `BCI2A_IDENTIFIABILITY_NOT_DETECTED`; the frozen 6/9 identifiability gate was not met, so no BCI2a GPU denoiser was trained.

BCI-IV-2b was used only as the pre-specified contamination-transfer contingency:

| protocol | MATCH-WRONG mean | positive | MATCH-POP mean |
|---|---:|---:|---:|
| same_session | +0.1823 | 8/9 | +0.0255 |
| cross_session | +0.1363 | 9/9 | +0.0047 |

BCI2b decision: `BCI2B_SUBJECT_IDENTIFIABILITY_DETECTED`. This development headroom cannot be relabeled as a BCI2a result.
