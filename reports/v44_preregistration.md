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

# V44-S2 addendum

```text
V44-S2 addendum — frozen before the first S2 submission.

RB-1 HONEST RE-BASELINING (co-primary, no new compute — recompute from S1 outputs):
  gain vs strongest subject-free route: MATCH_gated − NO_A0, mean/CI/positive count.
  The paper's gain claim leads with BOTH rows (vs POP anchor and vs NO_A0); the
  "bad anchor worse than none" finding (NO_A0 0.574 < POP 0.651) is reported as its
  own result and motivates the guard's fallback choice below.

OG OWNERSHIP GUARD (primary of this stage):
  Verification signal (deployment-legal in this class — query EOG+EEG are runtime
  inputs): fit a probe operator C_probe on the FIRST T_v seconds of the query stream
  (T_v = 30 primary; 10/60 sensitivity), same ridge/normalization contract as V41R.
  Mismatch score = Mahalanobis distance between the presented operator C_pres and
  C_probe in operator space, metric = the U0-b EB posterior covariance (fallback:
  row-space principal angles if U0-b unavailable on this branch).
  Threshold frozen from the M0 atlas: the score's 95th percentile under own-operator
  split-half resampling (no query outcomes used).
  GUARD DECISION: score > threshold -> a0 := 0 (the NO_A0 fallback — CHOSEN BECAUSE
  NO_A0 BEATS THE POP ANCHOR; the POP-anchor fallback row is reported descriptively).
  Endpoints (participant-first, 5000-draw bootstrap, Holm over {OG-1, OG-2}):
  OG-1 wrong-donor safety WITH guard: mean[RRMSE(WRONG_gated+guard) − RRMSE(best
       subject-free route NO_A0)] <= +0.005, and detection rate of wrong donors >= 0.9.
  OG-2 correct-donor cost: false-alarm rate <= 0.10 AND
       mean[RRMSE(MATCH_gated+guard) − RRMSE(MATCH_gated)] <= +0.005.

OR ONLINE REFINEMENT (secondary stage, preregistered):
  Warm-start recursive least squares on the query stream: C(t) updated causally from
  accumulated query EOG/EEG (forgetting factor frozen at 0.999; sensitivity 0.99),
  initialized at C_gated (warm) vs at C0 population (cold-pop) vs at zero (cold-zero);
  ORACLE row retained as ceiling. a0(t) = C(t)·e(t) fed to the frozen diffusion.
  Endpoints (descriptive + one test):
  OR-1 oracle-gap closure: fraction of the +0.244 ORACLE−MATCH gap closed by
       warm-RLS at t = end-of-record (test: warm-RLS − static MATCH_gated > 0, CI-low > 0).
  OR-2 calibration half-life curve (descriptive, figure-ready): warm − cold-zero gain
       as a function of deployment time t ∈ {10, 30, 60, 120, 240 s} — the value of
       calibration vs pure online adaptation over time. No monotone claim.
  OR-3 wrong-donor self-healing (descriptive): wrong-warm-RLS trajectory — does online
       data wash out a wrong initialization, and how fast.

NATURAL PANEL: since G4 validity passed for both arms, OG-1/OG-2 and OR-1 are also
  evaluated on the natural panel (attenuation/retention endpoints), reported with CIs.
No threshold changes after this commit. Sealed cohorts untouched.
```

# V44-S3 addendum

```text
V44-S3 addendum — frozen before any rescoring.

REGISTERED REPAIRS (both are null/feature corrections, not threshold tuning):
  R1 drift-calibrated null: threshold = 95th percentile of OWN support→query operator
     discrepancy across M0-atlas training owners (per feature family), replacing the
     split-half null. No query outcomes used.
  R2 drift-decomposed scores, two families evaluated separately:
     SPATIAL: principal angles between column spaces of C_pres and C_probe
              (largest angle primary; mean angle secondary)
     GAIN:    norm-ratio / per-row scale discrepancy (expected drift-dominated;
              registered as the negative-control family)

ENDPOINTS (same margins as S2, Holm over {OG-1', OG-2'}):
  OG-1' wrong-donor safety with the SPATIAL-score guard at the R1 threshold:
        mean[RRMSE(WRONG_gated+guard) − RRMSE(NO_A0)] ≤ +0.005 AND detection ≥ 0.90.
  OG-2' correct-donor cost: false-alarm ≤ 0.10 AND guard cost ≤ +0.005.
  ROC (descriptive, figure-ready): detection vs false-alarm curves for BOTH score
        families plus the S2 Mahalanobis score, with the utility-vs-operating-point
        curve (paired RRMSE of the guarded system across thresholds); natural-panel
        version (attenuation cost vs operating point).
  PREDICTION registered up front: SPATIAL separates (AUC materially above the S2
        score); GAIN does not. If SPATIAL also fails to reach OG-1'/OG-2', ownership
        verification from operator features is CLOSED for the likelihood leg and the
        papers report the complete two-family negative with the drift diagnosis.

No further ownership attempts after this stage regardless of outcome.
```
