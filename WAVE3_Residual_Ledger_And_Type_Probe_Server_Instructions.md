# WAVE3 — The residual ledger, the type probe, and the anomaly resolutions
### Server execution instructions (Claude Code) — merged five-work-package plan

Wave-3 attacks the four open problems (O1 conversion residual, O2 channel redundancy,
O3 prior-collapse cause, O4 W≈4τ² anomaly) under the ONE-PAPER directive: every funded
stage either repairs a sentence the flagship cannot currently ship, adds a measured row
to the residual ledger, or is a preregistered honest negative. Expected ~20–40 GPU-h
(hard cap 100); mostly CPU; zero sealed contact; structurally incapable of harming the
sealed +0.1537 anchor. Panel materials: `EEG_denoise_design_panel/wave3/` (forensics,
three designs, two judges — read as needed; this document is self-sufficient).

---

# 1. Workspace

```text
base    : codex/wave2-t1 tip 755ebe4
branch  : codex/wave3
worktree: continue in denoiseNet_wave2 or a fresh denoiseNet_wave3
compute : CPU/cpu-high dominated; GPU tranches ≤25 h through week 3
```

# 2. TIER 0 — ONE merged preregistration (commit BEFORE any computation)

Write `reports/wave3_preregistration.md` as a single frozen document containing ALL of:

```text
(i) CORRECTED RESIDUAL SEMANTICS (program-wide, immediate): the oracle-residual is
  reported in ADDITIVE semantics everywhere, with the total-semantics reading given
  alongside once; the W3 conversion fractions restate as 0.29–0.50 band; no document
  cites the old 0.50–0.66 or mixed-semantics numbers again.

(ii) UNITS-vs-COMPOSITION DECISION TREE for O4/P2 (the 0.134 degeneracy fix):
  STEP 1 = B0 CODE READ FIRST: read eb_transfer_v43.py and the wave-2 shared-layer
  script; establish the literal λ estimator (the λ = τ²/(τ²+within/4) form is an
  INFERENCE — its sole support is the 0.4993 coincidence — label it
  inference-pending-B0 until the code says otherwise) and whether the gate's "within"
  is per-window variance or block-mean variance.
  STEP 2 = the T7 2×2 FACTORIAL {split granularity: window-random vs block-level} ×
  {stratification: none vs type-stratified}, both accounts' DISTINCT predictions frozen:
    units account:       W(block-level, unstratified) ≈ 0.134 AND
                         W(window, stratified) ≈ W(window, random) ≈ 1.09
    composition account: W(window, stratified) collapses toward 0.134
  The factorial, not any 0.134 landing, is the discriminator. If units wins: P2
  dissolves as a units lesson, O4 restates per-window (mundane candidate: 92 ridge
  parameters per ~3.7-s window), the parsimony narrative is dropped, and A1 survives
  only as the natural-data ceiling question. Composition-term closure check: W_random −
  W_strat reproduced within ×2 by mixture-fraction variance × T1's measured separation.

(iii) UNIFIED TYPE GATE (one gate for the one crux): natural-data T1 census gate =
  TSR ≥ 2 AND bootstrap CI-low > 1.3 AND ≥10/15 subjects AND classifier κ ≥ 0.8 AND
  mixture non-degeneracy (both types ≥15%). The ≥8/15 variant is labeled sensitivity
  only. κ < 0.8 → INCONCLUSIVE (not NO-GO).

(iv) INVALID-INSTRUMENT RULINGS (from the hostile audit, binding):
  T1-pre (banked per-window residual typing) is DESCRIPTIVE PLUMBING ONLY, |e|²-
  normalized, NO kill authority — on single-injection semi-sim pairs type-independence
  holds by construction and spurious type-structure appears via |e|-scaling.
  The S0b query-fitted typed-vs-pooled RRMSE probe on banked pairs is DELETED for the
  same reason. The single RRMSE-unit oracle instrument is Panel-T (below), built once,
  jointly owned, attribution-only.

(v) CLOSURE LEDGER (binding rulings): type-indexed STATIC operators are LEGAL
  (indicator-expansion identity: type-indexed 46×2 ≡ static 46×4 on [e·1_blink;
  e·1_sacc] — same legality class as the confirmed +0.143 channel). RED LINES:
  ψ (the type classifier) thresholds frozen GLOBALLY in this prereg, never tuned
  per-subject or on query data; starved/abstained types degrade bit-identically to the
  incumbent; no output of any diagnostic oracle (T2, Panel-T) converts into a deployed
  within-query estimator this wave; a T7/Pack-B explanation of the DT-Gibbs G0 death
  does NOT resurrect the semi-blind family (new registered design required); ONCE
  reopens nothing (U-1 was a factorial verdict on naive stacking; the naive stack is
  retained as comparator); M-B does not bar ocular type-resolution.

(vi) DISJOINT LEDGER-ROW DEFINITIONS (F6): the residual ledger rows {delivered,
  bookkeeping, gate-shrinkage, estimation-noise, readout, family, fluctuation, drift,
  unattributed-remainder} are defined as a disjoint decomposition with stated
  covariances (B2 and B4 both derive from λ̂ — partition, do not double-count;
  B3 overlaps A1's ceiling term — assign by the frozen rule).

(vii) ONCE noise-corrected decision rule (F5): the Stage-0 primary statistic is the
  deficit decomposition (≥60% of the joint-vs-best deficit along the shared subtraction
  span WITH excess removal ⇒ bookkeeping conviction; orthogonal-variance-dominated ⇒
  noise conviction) — no "impossibility" rhetoric; compute a CI on the banked −0.0072
  before asserting over-subtraction.

(viii) A4 REFERENCE-CHANNEL ERROR is added to the attribution tree (EOG measurement
  noise + neural crosstalk caps every e-regressing arm) with a ledger row; its clean
  instrument (Eye-BCI optical-vs-EOG) is priced but deferred.

(ix) Pack-A hard cap 25 GPU-h; A2' sandbox runs ONLY if A1' inconclusive AND ≥1 density
  fingerprint fired at A0. TROCA S2 (Eye-BCI) deferred out of wave 3; S3 retraining
  unfunded. Statistics everywhere: participant-first n=15, 5000-draw bootstrap,
  sign-flip, Holm within families; C05/C08 wording; PV-2/q99 guards on any new arm.
```

# 3. Work packages (run per the sequencing in §4)

**WP-A LEDGER CORE (CPU)**: B0 code read → T0 bookkeeping audit (corrected R* with CIs
per panel; δ_conv = ρ̂ − delivered per panel) → B1 zero-free-parameter transform test
(GO: |λ_pred−λ̂| ≤ 0.03 per flagship cell AND slope CI ∋ 1 cross-panel) + B1.5
sympy/Monte-Carlo estimator verification → T7 2×2 factorial (with stratified-D
recompute — any restored-spine ρ̂ statement is conditional on it) + T8 per-coefficient
recompute. Spine restoration (ρ̂ language back into plan vocabulary) ONLY on full
cross-panel PASS. Include the descriptive post-hoc λ̂ read of the already-opened,
sha256-frozen C-1 logs (legal, descriptive only).

**WP-B TYPE PROBE (CPU + gated GPU)**: T1 natural-data typed census on the EOG channels
(blink vs vertical-EM within the VEOG column is the true target — the 46×2 family
already separates VEOG/HEOG columns; the classical claim survives only within-column):
one frozen classifier ψ (morphology thresholds global), typed operator fits per subject,
separation² / W_type, the unified gate (iii). T2 nested variance decomposition. T6
nested family-ceiling ladder on natural queries (indicator-linear → rank-3 → FIR-lagged
→ amplitude-gain → kernel ridge; CV-residual + EOG-band attenuation + retention-guarded
— this is the family-misspecification adjudicator). On T1 GO: build Panel-T ONCE
(typed-injection semi-sim rebuild, separation fixed at T1's measured value, shuffled-type
control + original-panel expected-null control; ≤10 GPU-h; numbers are attribution-only,
never headline). On Panel-T oracle slice ≥ +0.03 CI-low > 0: TROCA S1 deployment stage
(15–30 GPU-h, inference-only on frozen V44 checkpoints: TR-1 typed-anchor vs pooled
incumbent ≥ +0.02 CI-low > 0; TR-2 wrong-donor reported verbatim as the expected 5th
identity-blind replication; bit-identity assert for starved types; shuffled-type must
erase the gain; TR-4 G4 non-degradation vs banked +2.46 dB/0.843; ITT abstention).

**WP-C ONCE STAGES 0–2 (CPU, ≤5 GPU-h)**: Stage 0 autopsy of U-1 (CI on −0.0072;
deficit decomposition per (vii); both-fire applicability domain from banked abstention
rates) → Stage 1 algebra + simulation ledger (the no-double-count projection identity
in house style; the fitted cross-term MUST retrodict additivity 0.596 AND joint−best
−0.0072 before any new arm runs) → Stage 2 orthogonalized composite on banked windows
(hard gates: bit-identity on single-channel strata; P1 additivity ≥ 0.90; P2
non-inferiority ε = 0.002; superiority claims only at CI-low > 0). Branch B (additivity
≈1, joint ≈ best single) is the modal, fully publishable outcome: the two channels tap
one shared ocular budget.

**WP-D COLLAPSE-ID (Pack A, hard-capped 25 GPU-h)**: A0 entry gate (reproduce the dev
U-ratio < 0.95 with CI excluding 1, ≤5 GPU-h; STOP on fail — finding 7 softens to a
single-run observation) → A0 CPU fingerprints (masking-frequency correlation, amplitude
ladder, prevalence tables) → A1' guidance-weight sweep (5–10 GPU-h; C4/guidance
confirmed ⇒ skip sandbox arms) → conditional A2'/A3' per (ix). Attribution requires
in-vitro reproduction ≥50% magnitude AND banked-fingerprint match. This is a paper
OBLIGATION: finding 7 currently cites a mechanism falsified in vitro.

**WP-E CHEAP LEDGER ROWS**: T4 exact gate-shrinkage recompute (certain value; may yield
a free deployment note: shrink-to-zero vs shrink-to-pop, NO_A0-consistent). T3
oracle-by-readout (≤5 GPU-h banked inference; |residual_DIFF − residual_LIN| ≤ 0.03
exonerates the sampler; the surprise branch triggers the C08-compliant "LINEAR
delivery, diffusion UQ" rewording).

# 4. Sequencing and decision points

```text
W0   : Tier-0 prereg commit → B0 code read → T0 → ONCE Stage 0 → Pack-A CPU
       fingerprints → verify T1-pre bankedness (descriptive only regardless)
W1-2 : B1/B1.5 → T7 factorial + T8 → T1 census + T2 + T6 (CPU) → T4 → ONCE Stages 1-2
       → Pack-A A0 gate (≤5 GPU-h)
DP1  : (a) units-vs-composition verdict per the frozen tree; (b) T1 GO/NO-GO/
       INCONCLUSIVE (NO-GO at adequate κ ⇒ A1 closed as a first-class negative:
       every measured ceiling certified family-final against the classical prior);
       (c) ONCE branch call; (d) A0 verdict
W2-3 : T3, T6 completion, A1' sweep, Panel-T (only on T1 GO)          [≤25 GPU-h]
DP2  : Panel-T oracle-slice gate decides TROCA S1; A1' decides Pack-A tail
W3-4 : TROCA S1 deployment stage (only on DP2 GO)                      [15-30 GPU-h]
W5   : LEDGER ASSEMBLY — the disjoint residual-decomposition table with CIs and prereg
       citations per row, annotating the M0 matrix; corrected conversion fractions
       adopted program-wide. This figure is the wave's guaranteed deliverable.
```

Stop points: after DP1 (report all verdicts), after DP2, and after the ledger assembly.
Report every decision JSON verbatim. Overrun past 100 GPU-h triggers re-prioritization,
never silent spend.

# 5. Prohibitions

```text
no sealed contact (MobileBCI-8 spent; BrainID Day-200 / PhysioMotion-10 / SHU stay
  sealed; the C-1 log read is descriptive on already-frozen artifacts)
no modification of any banked artifact or frozen closure
no per-subject or query-side tuning of ψ; no deployed within-query estimators from
  diagnostic oracles; no semi-blind resurrection; no prior-axis rebuilding
no new mechanism families; TROCA S2/S3 out of wave 3
no writing-round text generation (documentation and ledgers only, per the operator)
```

# 6. Kickoff prompt

```text
Read WAVE3_Residual_Ledger_And_Type_Probe_Server_Instructions.md in full and execute
it. Create branch codex/wave3 from codex/wave2-t1 tip 755ebe4. Commit the single merged
Tier-0 preregistration FIRST (it contains the corrected residual semantics, the
units-vs-composition decision tree with B0-code-read-first, the unified type gate, the
invalid-instrument rulings, the closure ledger, the disjoint ledger-row definitions,
and all caps). Then run WP-A/WP-C/WP-D week-0 items, the week-1-2 CPU battery, and
write Decision Point 1; proceed through the gated GPU tranches and Decision Point 2;
finish with the ledger assembly. Slurm only (CPU/cpu-high; GPU tranches ≤25 h through
week 3); no sealed contact; no verification ceremonies; report all decisions verbatim
and stop at each declared stop point.
```
