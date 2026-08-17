# IRIS F-stage preregistration — F1 (MobileBCI diffusion-path fight) + F2 (EEGEyeNet dual-reading fight)

Committed BEFORE any F-stage execution. Inputs: adopted gate BINARY_NOA0FB (P1R,
55ca52e), typed family qualified additive-only (T2). Decision JSONs →
`results/iris/f1/f1_decision.json`, `results/iris/f2/{f2a_decision,f2b_decision}.json`.
Program conventions throughout (ridge 0.05, bootstrap 5000 seed 420,
participant-first primary, ITT).

## F1 — does the adopted fallback transfer to the deployed diffusion class? (D1)

**Machinery (frozen).** V44-S1 evaluation verbatim: the 15 frozen fold×seed
checkpoints, `TransferEpisodeSampler(..., "test", seed+3).sample_balanced(8)`,
`sample_bank_eog` DDIM sampling with the V44 `noise_seed(fold, seed)` convention,
`paired_metrics` → `rrmse_temporal`. Arms:

| Arm | a0 input | signature |
| --- | --- | --- |
| MATCH_gated | `C_gated @ drive` (incumbent deployed) | sig_gated |
| MATCH_NOA0FB | hard-gated cells: **0**; else identical to MATCH_gated | sig_gated |
| NO_A0 | 0 | sig_gated |
| POP | `C0 @ drive` | sig_pop |

Correctness guard (frozen): on non-hard-gated cells MATCH_gated and MATCH_NOA0FB have
bit-identical inputs and the same noise seed → outputs must be bit-identical; any
difference is an instrument defect and stops the run.

**Gates (frozen).**
- **F1-primary (cell-level, where the change lives):** paired contrast
  RRMSE(MATCH_gated) − RRMSE(MATCH_NOA0FB) over hard-gated cells, pooled across
  folds/seeds. WIN = bootstrap CI-low > 0; TIE = CI spans 0; LOSS = CI-high < 0.
  Reported as the D1 adjudication of IRIS-vs-incumbent on this panel: IRIS's only
  surviving point-estimate delta IS the fallback (inflation retired, drift off,
  Class-E typing invalid) — the fight is exactly this contrast, and a TIE means the
  incumbent survives as IRIS's point system on MobileBCI.
- **F1-anchor (preservation):** MATCH_NOA0FB − NO_A0, participant-first mean with CI;
  must remain positive with CI-low > 0 (banked incumbent dev anchor: +0.1428
  [15/15]); magnitude reported beside the banked value, never edited.
- **F1-natural (descriptive):** V44 `_natural_metrics` attenuation/retention for both
  MATCH arms on hard-gated cells' natural windows.

Budget: ≤ 20 GPU-h (single job, incremental per-fold-seed outputs, resubmit skips
completed units). GPU partitions A100/H100/L40S by availability.

## F2-a — dots dual-reading fight (D2), linear class, CPU

Both readings preregistered per the judge warning; reported regardless. Subtraction
arms fit on FIT thirds (repaired gaze encoding), applied on EVAL thirds; frontal =
the K3/W frozen 8-channel anterior block.

| Arm | Reference design |
| --- | --- |
| INCUMBENT_NATIVE | static [VEOG, HEOG] × 5 lags (10 regressors) |
| INCUMBENT_SAMEREF | static [VEOG, HEOG, gaze-x, gaze-y, pupil] × 5 lags (25; the same-reference control: identical information, no typed structure) |
| IRIS_TYPED | STATIC ∪ TYPED (35; T2's additive design) |

**Endpoints (frozen).**
- E1 artifact-window attenuation: 10·log10(var(y)/var(x̂)) on the top-20%
  VEOG-energy EVAL windows, frontal block.
- E2 low-EOG observation retention (C08 naming): the V44 `_natural_metrics` retention
  definition VERBATIM (ported constant-for-constant), computed on the bottom-50%
  VEOG-energy EVAL windows.
- E3 λ-wave exogeneity HARD GATE: retention ≥ **0.84** of the posterior block (8 most
  posterior channels by chanlocs X) in post-saccadic [0, 300 ms] windows, per arm. An
  arm failing E3 is INVALID regardless of E1/E2 (the OTTO hazard bar).

**Verdicts (frozen, each reading).** Conditional on both arms passing E3 and on
retention non-inferiority (E2 delta ≥ −0.02): paired per-recording E1 delta,
participant-first CI → WIN / TIE / LOSS by CI vs 0.
- Same-reference reading: IRIS_TYPED vs INCUMBENT_SAMEREF (isolates typed STRUCTURE).
- Native reading: IRIS_TYPED vs INCUMBENT_NATIVE (deployment-fair).

## F2-b — antisaccade-dev typed increment (sealed-fight qualifier), CPU

The sealed panel has no continuous gaze; typed drives there are event trains only.
T2's machinery on the 28 antisaccade dev recordings with TYPED = {blink train ⊗
template, signed saccade train ⊗ template} (no gaze channels), same thirds, same
rich-window definition (top-20% VEOG energy), same null control (circular shift ≥5 s,
seed 420), same gates: participant-first CI-low(Δ_inc) > 0 AND mean ≥ 0.05 AND
CI-low(Δ_inc − Δ_null) > 0. VEOG on antisaccade uses the frozen periocular derivation;
the 26/28 periocular-interpolated recordings are INCLUDED here (the reference just
degrades to its interpolated reconstruction — deployment-realistic) with the
interpolation status carried per recording and a 2-recording intact-only companion.

**Consequence rule (frozen).** F2-b PASS → the sealed-fight plan (separate prereg,
operator sign-off) carries a typed arm. FAIL → the typed leg is dead on the sealed
panel class; any sealed opening proposal shrinks to incumbent-class confirmation and
says so.

## Amendment F-1 (committed before execution) — F2-R spatially-gated subtraction

**Motivation (banked F2-a).** Every unrestricted arm fails E3 (0.354-0.404 vs 0.84,
incumbent included): the exogeneity failure is family-independent. The program's
abstention mechanism, applied spatially, is the registered repair: a channel receives
subtraction only where its fit is validated out-of-sample BEFORE evaluation.

**Per-channel gate (frozen; FIT data only, EVAL untouched).** Split FIT into its two
constituent thirds. For each arm and target channel c, fit on one third and validate
on the other, both directions. Channel c is SUBTRACTION-ELIGIBLE iff in BOTH
directions: (a) validated prediction r² >= 0.10, and (b) validated per-channel
post-saccadic retention >= 0.84. Abstained channels receive zero subtraction
(fail-closed). The deployed EVAL estimate uses the full-FIT fit restricted to
eligible channels.

**Endpoints and verdicts.** E1/E2/E3 and both readings exactly as F2-a, computed on
the gated arms; per-arm eligible-channel counts (frontal/posterior) reported. E3 is
still measured on the full posterior block on EVAL — if it passes because posterior
channels abstained, that is the fail-closed mechanism working and is reported as
such, not as a subtraction-validity claim. If any arm STILL fails E3, that arm's
family is closed for this panel with no further repairs.
