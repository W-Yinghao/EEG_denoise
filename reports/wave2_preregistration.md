# WAVE2-T1 preregistration

```text
WAVE2-T1 preregistration — frozen before submissions. The deep2 design files are not on
disk; the WAVE2 instruction file's numerical gates are the frozen source of truth, and
the operational definitions below are REGISTERED HERE (fixed before any computation).

LANGUAGE RULE (from the hostile audit): the delivered/ceiling conversion band is 0.3-0.5
  (residual semantics unified: state per-leg whether the oracle number is additive
  residual or total, and compute both readings); D ∈ [τ², 2.4τ²] with D≈τ² labeled the
  optimistic endpoint. The AUC 0.70-0.80 retrodiction is a consistency check (contains an
  unstated low-dimensionality assumption), not a confirmation.
NOT RUN (superseded/barred): the P10 coherence-guard experiment; the importance-
  reweighting prior probe.

SHARED LAYER (Tier-1, CPU), registered estimators, MobileBCI dev cohort (per-fold then
  pooled), Klados/BCI2b where computable:
  τ²  = mean squared deviation of dev owners' 120-s operators around the cohort mean
  W   = mean within-cell 4-block scatter of the 120-s fit (V43 semantics)
  D   = own support(120-s)→query(Qgen) operator discrepancy, BOTH readings reported:
        D_raw = mean‖C_support − C_query‖²_mean (upper/total reading) and
        D_deb = max(0, D_raw − W/4 − W_q/4) (debiased/additive reading)
  Σ_drift = entrywise variance over dev cells of (C_query − C_support)
P2 LEDGER TEST: λ_pred = τ²/(τ² + W/4) must be within ±0.03 of the measured λ̂ cohort
  mean (0.88-0.92 banked), AND at least one conversion reading (additive
  0.143/(0.143+0.244)=0.369; total 0.143/0.502=0.285 recomputed from the shared layer's
  own numbers where possible) must land in [0.3, 0.5]. PASS/SOFT-FAIL; SOFT-FAIL
  downgrades Σ_drift to an empirical object everywhere and strips ρ̂-prediction language
  from sealed plans.
P1 SAME-BLOCK REPLAY (banked V44-S1 checkpoints, 5 cells seed 20261201): SAMEBLOCK arm
  = MATCH_gated with C fit on the first 60 s of the Qgen interval (same chronological
  block as the generator; same ridge/normalization); conversion fraction per variant
  = (NO_A0 − arm)/(NO_A0 − ORACLE), participant-first. ADJUDICATION: sameblock fraction
  ≥ 0.75 ⇒ drift binds (OPERA-E2 customer evaporates); ≤ 0.55 ⇒ prior/interface binds
  (DT-Gibbs prize shrinks); between ⇒ mixed attribution, both stated.
P7 ANCHOR DOSE-RESPONSE (banked, same cells): C(α) = (1−α)·C_gated + α·C_wrong_gated,
  α ∈ {0, 0.25, 0.5, 0.75, 1}; harm(α) = RRMSE(α) − RRMSE(0). PREDICTION (manifold
  displacement): superlinear growth — harm(0.5) < 0.5·harm(1.0) with bootstrap CI-high
  of [harm(0.5) − 0.5·harm(1.0)] < 0.
DRIFT-WIDENED W4 REPLAY (banked, 5 cells seed 20261201, K=8 equal-weight chains):
  C ~ N(C_gated, Σ_post ⊕ Σ_drift) (diagonal sum), no Gibbs; endpoints = coverage
  50/80/90 vs W4's 0.271/0.440/0.516 (bands [0.35,0.65]/[0.65,0.90]/[0.80,0.97]) and
  CRPS vs 0.1529.

MOKA M-A (CPU; STEP ZERO = verify on disk that the frozen v4/v5 MobileBCI motion blocks
  + synchronized IMU exist and are readable, AND pull the v4/v5 closure wording from the
  repo branches to confirm it covers temporal-support conditioning only): artifact-band
  (0.5-15 Hz) IMU-predictable EEG energy fraction per subject/protocol via ridge
  regression of EEG on the 27 IMU channels over motion blocks (R² in-band, split-half
  validated). GO to M-B iff median fraction ≥ 0.10. M-B (CPU): own-vs-population
  motion-operator headroom (V19-style paired construction on motion blocks). GO iff
  mean ≥ +0.02 with CI-low > 0 in ≥1 protocol.
OPERA A0 (CPU): corpus/corruption audit with the e-EXOGENEITY BOUND AS A HARD GATE:
  neural leakage into the EOG reference estimated as the R² of the bipolar EOG explained
  by posterior-quadrant EEG channels in LOW-EOG windows (dev cohort); HARD GATE:
  leakage R² ≤ 0.15 (frozen here). Eye-BCI optical reference noted as the only clean
  escape. A1 (15-25 GPU-h): synthetic double dissociation — a clean-window-selected
  prior shows the U° energy deficit AND over-subtraction; an ambient/EM-trained prior
  shows neither. GO iff the dissociation is clean in BOTH directions.
DT-GIBBS G0 (CPU): drift prior N(C_support, Σ_post ⊕ Σ_drift); prior-predictive 80%
  coverage of held-out own-query operators must land in [0.70, 0.90] (entrywise,
  participant-first). G1 (5-10 GPU-h): one-sweep Gibbs C-step on frozen V44-S1
  checkpoints with TWO-SIDED equivalence gates sized against GB-1's +0.03 margin:
  |clean-pass Ĉ bias| ≤ 0.015 and |Ĉ − C_true| bias ≤ 0.015 (bootstrap CI within the
  equivalence band); the C-step bias direction is preregistered TWO-SIDED (impure prior
  may inflate Ĉ → over-subtraction). NO-GO closes the semi-blind family.
THRESH T0 (CPU): atlas-calibrated task priors; d_eff = participation ratio of the
  operator-deviation covariance spectrum; closed-form reference curves under the
  measured anisotropy; FROZEN RULE: the conditional-harm prediction survives on paper
  iff d_eff ≤ 20 (of 92 ambient); otherwise the T1a demo shrinks to transition-only.
  T1a (10-12 GPU-h): operator-space ICL grid — 4-size attention-model ladder trained on
  synthetic operator-inference episodes with atlas-calibrated priors (context length
  grid), transition metric = context-use vs prior-use crossover location vs d_eff;
  falsification branch: no size-dependent transition ⇒ threshold law rejected on this
  family.
TIER-2 GATES (frozen now, run later only on GO): DT-Gibbs G2 (40-80 GPU-h; per the
  sealed ledger its confirmation route is SHU Day-4/5, since the M35 opening has
  occurred); THRESH T1b (40-50; only if T1a fires AND the standalone paper is wanted);
  OPERA A2 (60-120; behind A1); MOKA M-C (60-120; behind M-B GO + closure-wording
  verification). Priority if compute or attention binds: G2 > T1b > M-C > A2.
Statistics: participant/record-first, 5000-draw bootstrap; Holm within each design's
  gate family. No sealed reads anywhere in WAVE2-T1. No threshold changes post-commit.
```
