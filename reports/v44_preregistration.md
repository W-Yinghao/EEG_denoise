# V44 preregistration

```text
V44 preregistration — frozen before first Slurm submission.

INFORMATION BOUNDARY: in this deployment class the two registered bipolar query-EOG
regressors (VEOG, HEOG) are runtime inputs at inference. The Qgen-fitted operator remains
evaluator-only (ORACLE arm). Sealed participants are never read.

OPERATOR ARMS (subtraction and conditioning both use the V43-frozen EB gate; lambda rule,
hard gate, and clamp contract unchanged — no retuning):
  C_gated      own 120-s support transfer, EB-gated toward the population operator
  C0           recipient-excluded fold-train population operator (strong POP)
  C_wrong      unseen wrong-donor 120-s transfer, ungated
  C_wrong_g    wrong-donor transfer gated with ITS OWN lambda
  C_query      Qgen-fitted operator (ORACLE, evaluator-only)

V44-S0 (CPU subtraction probe, paired panel, full-window temporal RRMSE vs clean x,
participant-first n=15, 5000-draw bootstrap):
  G0-1 (GO rule for S1): gain = RRMSE(y - C0*e) - RRMSE(y - C_gated*e)
        GO iff mean >= +0.010 AND bootstrap CI-low > 0.
  G0-2 gate safety: RRMSE(y - C_wrong_g*e) - RRMSE(y - C0*e) <= +0.010.
  G0-3 (descriptive): ungated wrong-donor harm; ORACLE ceiling row.
  G0-4 (descriptive, natural windows): attenuation (dB), EOG coherence reduction,
        low-EOG observation retention for the C_gated vs C0 subtraction arms.
  If G0-1 GO: proceed directly to S1 without waiting for the operator.
  If NO-GO: stop, report, no training.

V44-S1 (EOG-guided diffusion, retrained; participant-first n=15):
  Primary G1: within the diffusion system, MATCH_gated - POP utility
        mean[RRMSE(POP arm) - RRMSE(MATCH_gated arm)] > 0 with CI-low > 0.
  G2 controls: WRONG_gated within +0.005 of POP; ungated WRONG harmful (descriptive);
        SHUFFLED-EOG (temporally shuffled query EOG in a0) markedly worse than MATCH_gated;
        NO_A0 (a0 = 0) reproduces the conditioning-class behavior (descriptive bridge row);
        ORACLE - MATCH_gated reported as the residual estimator gap.
  G3 positioning (descriptive, C05-compliant): capacity-matched DET-EOG twin and
        LINEAR-EOG subtraction rows; wording "competitive", no superiority claim either way,
        in either direction.
  G4 natural validity bar (frozen): a natural claim for an arm requires
        attenuation > 0 dB AND low-EOG observation retention >= 0.75 for that arm;
        if met, MATCH_gated - POP natural utilities reported with CIs; else descriptive only.
  Statistics: Holm over {G1, G2-wrong-gated, G2-shuffled}. No post-hoc endpoints.
```
