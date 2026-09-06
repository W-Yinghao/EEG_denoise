# V26 final development diagnosis

## Outcome

V26 is engineering-valid and preserves a paired-development subject-context signal. The matched one-step refiner is the stronger point estimator, but it is a competitive positioning and mechanism control—not a diffusion-retention gate. Natural artifact–preservation validity is the primary interpretive criterion.

The selected positive-noise operating point was `sigma_start=0.05`, DDIM 10 steps, K=1, natural fraction 0.30. `sigma=0` remained the exact deterministic reference and was excluded from diffusion candidates.

## Paired development

Positive utility means the first method is better.

| Contrast | Mean | 95% bootstrap CI | Positive |
|---|---:|---:|---:|
| one-step vs V25 DET | -0.000047 | [-0.002055, 0.001945] | 8/15 |
| SDEdit vs matched one-step | -0.003575 | [-0.005380, -0.001642] | 2/15 |
| SDEdit support vs PopSDEdit | +0.009143 | [0.001233, 0.018317] | 10/15 |
| SDEdit MATCH vs WRONG | +0.009850 | [0.004789, 0.015246] | 12/15 |
| one-step MATCH vs WRONG | +0.011046 | [0.004694, 0.017821] | 10/15 |

Mean clean temporal RRMSE was 0.70613 for V25 DET-MATCH, 0.70618 for CalibRefineDET-MATCH, and 0.70975 for CalibSDEdit-MATCH. SDEdit therefore retains support sensitivity but does not improve the matched one-step point estimate.

## Natural development

CalibSDEdit-MATCH had mean remaining ratio 1.00771, attenuation 1.36311 dB, preservation 0.80336, PSD distortion 0.38653, and covariance distortion 0.23111. The support contrast against PopSDEdit was mixed:

| Contrast | Mean | 95% bootstrap CI | Positive |
|---|---:|---:|---:|
| SDEdit support artifact utility | +0.009705 | [-0.022409, 0.036629] | 12/15 |
| SDEdit support preservation utility | -0.007522 | [-0.014191, -0.000750] | 5/15 |
| SDEdit vs one-step artifact utility | -0.019899 | [-0.033075, -0.007021] | 1/15 |
| SDEdit vs one-step preservation utility | -0.008492 | [-0.011888, -0.004594] | 1/15 |

The natural classification is `preservation_concern`: support-conditioned SDEdit shows an artifact-direction signal relative to its population refiner, but its preservation cost is systematic and it is inferior to the matched one-step refiner on both natural axes. This natural evidence—not a strict paired `DIFF > DET` rule—drives the interpretation.

## Forensic and governance

The V25 learned-basis target is coordinate-unstable: an equivalent rotation changed the coefficient target by 1.60667 relative units while leaving the sensor artifact invariant to 1.33e-15. V26 therefore used a fixed 46-channel sensor artifact target.

All 60 Round-B model cells and all 15 paired/natural evaluation cells were retained. End-to-end evaluation of the 11-method inference bundle averaged 22.89 ms per window. Query-EOG, query-operator, event, and sealed inference reads were zero. A-track and `taas_submission/**` were unchanged; no confirmation or manuscript compilation occurred.

## Classification and next route

- Engineering: `valid`
- Subject context: `paired_signal_preserved`; natural signal remains conflicted
- One-step: `equivalent_to_base_det`
- Diffusion: `one_step_better` as competitive point-estimation positioning
- Natural: `preservation_concern`
- Next: **D. add lightweight energy refinement**

The next experiment should be a small, explicitly bounded energy refinement aimed at artifact–neural overlap. It must preserve the deterministic/support evidence and must not expand into a large routing, rollback, or operator-portfolio system.

These are development-only findings. They do not establish confirmation, SOTA, deployment safety, exact posterior inference, permanence, cross-montage validity, or clinical validity.
