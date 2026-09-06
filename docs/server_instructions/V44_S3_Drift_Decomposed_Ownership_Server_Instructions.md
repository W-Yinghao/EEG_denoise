# V44-S3 — Drift-decomposed ownership rescoring (CPU-only micro-stage)
### Server execution instructions (Claude Code)

S2's ownership failure has a diagnosed, mechanism-backed cause, and both fixes are
protocol repairs in the v19→v20 tradition (invalid null → valid null), not outcome
tuning: (1) the frozen threshold was calibrated on support split-half variability, but
the operative null is support→query DRIFT — the correct null distribution exists in the
M0 atlas (training owners have both support and Qgen operators); (2) the Mahalanobis
score mixes gain/temporal drift with spatial wrongness, while O1-V21 independently
established that the SPATIAL structure is what transfers across time
(SPATIAL_TRANSFER_VALID / temporal coefficient drifts). Hypothesis, preregistered before
any rescoring: own-drift is gain-dominated; wrong-donorship is spatial — a column-space
principal-angle score is drift-robust.

Everything runs on frozen stored arrays (S2 probe fits, S1/S2 arm outputs). The guard is
a router between already-sampled outputs (MATCH_gated vs NO_A0 per record), so all
endpoints recompute on CPU. Zero GPU.

---

# 1. Workspace

```text
continue : denoiseNet_rgcc_eog_v44, branch codex/rgcc-eog-v44, tip e083542
compute  : CPU / cpu-high only
```

# 2. Preregistration addendum (append BEFORE running)

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

# 3. Execution

Single CPU job (`v44_stage3.sbatch`, partition CPU/cpu-high): load stored S2 probe/presented
operator arrays + M0 atlas support/Qgen operator pairs; compute both score families and
the R1 thresholds; re-route the frozen S1/S2 outputs under each guard decision; recompute
OG-1'/OG-2', the three-family ROC, and the operating-point utility curves (paired +
natural). Write `results/rgcc_eog_v44/stage3/decision.json` + `reports/v44_stage3.md`.
Commit and stop; report verbatim.
