# V44-S2 — Ownership guard, honest re-baselining, online refinement
### Server execution instructions (Claude Code)

V44-S1 delivered the arc's first live subject-aware system gain (+0.220, 14/15, natural-
valid) with two exposures this stage closes: (1) the reliability gate does not stop a
well-estimated WRONG operator in the likelihood class (G2-wrong-gated +0.038 > +0.005;
same on BCI2b per V43-S3c) — the anchor channel needs OWNERSHIP verification, not just
reliability; (2) the POP anchor is a weak baseline (NO_A0 0.574 beats POP-anchor 0.651),
so the N3 honesty rule applied to ourselves requires re-baselining the gain against the
strongest subject-free route. Everything here is inference-time on the frozen V44-S1
checkpoints — no retraining.

---

# 1. Workspace and Git

```text
continue : worktree denoiseNet_rgcc_eog_v44, branch codex/rgcc-eog-v44, tip 1ae2980
```

Harness unchanged. V44-S1 checkpoints and results read-only; new results under
`results/rgcc_eog_v44/stage2/`.

---

# 2. Preregistration addendum (append to reports/v44_preregistration.md BEFORE submissions)

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

---

# 3. Execution

```text
RB-1     : CPU re-aggregation of frozen S1 outputs (no new sampling)
OG arms  : inference-only on frozen checkpoints — WRONG_gated+guard (14 donors x
           recipients as in S1), MATCH_gated+guard, fallback rows; common noise,
           S1 seed convention; any GPU partition, --time=04:00:00, array 0-14
OR arms  : causal RLS is CPU; the diffusion re-sampling per time-grid point is GPU
           inference — batch the a0(t) variants into the same array jobs
aggregate: CPU; decision JSON results/rgcc_eog_v44/stage2/decision.json
           {RB-1 rows, OG-1, OG-2, OR-1, natural versions, half-life table}
report   : reports/v44_stage2.md
```

Budget: ~10-20 GPU-hours total.

Stop after the decision JSON. Report RB-1 (both gain rows), OG-1/OG-2 verdicts, the
oracle-gap closure fraction, and the half-life table verbatim. These rows complete the
likelihood leg for both papers (V44 gain leg + flagship W4/unification), and the guard
generalizes to the flagship's transport leg (same construction with transport probes) —
that port happens in M3-5, not here.

Prohibitions: no retraining; no threshold changes post-commit; no sealed reads; the S1
G1/G2 verdicts are frozen (S2 adds rows, revises nothing).

---

# 4. Kickoff prompt

```text
Read V44_S2_Ownership_Guard_Server_Instructions.md in full and execute it. Continue on
codex/rgcc-eog-v44 (tip 1ae2980). Commit the preregistration addendum BEFORE any
submission. Recompute RB-1 from frozen S1 outputs; freeze the ownership-guard threshold
from the M0 atlas split-half distribution; run the OG and OR inference arms on the frozen
V44-S1 checkpoints (common noise; any GPU partition, 4 h cells); evaluate paired and
natural endpoints; write the decision JSON and report; commit and stop. Slurm only; no
verification ceremonies; no sealed reads; no retraining.
```
