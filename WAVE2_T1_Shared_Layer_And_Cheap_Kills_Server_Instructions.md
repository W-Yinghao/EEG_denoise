# WAVE2-T1 — Sealed ledger, shared atlas/drift layer, and the cheap-kill battery
### Server execution instructions (Claude Code) — Wave-2 Tier 1 + first GPU wave

Wave-2 funds four new mechanism families (DT-Gibbs, THRESH, MOKA, OPERA) through their
cheap-kill gates. This stage builds the shared quantitative layer once, runs every
CPU-priced adjudicator, freezes ALL Tier-2 preregistrations, and then runs the three
small first-wave GPU probes. Nothing here touches any sealed cohort; everything runs
beside M35 without contention. Portfolio doc: `EEG_denoise_wave2_portfolio.md`; design
details: `EEG_denoise_design_panel/deep2/design-{gibbs,tide,motion,ambient}.md`;
mechanism account + corrections: `deep2/{mechanism.md, judge-hostile.md}`.

---

# 1. Workspace

```text
base    : codex/flagship-m13 tip f9d5dc2 (or codex/flagship-m35 tip if M35 has landed)
branch  : codex/wave2-t1
worktree: /home/infres/yinwang/denoiseNet_wave2
compute : CPU/cpu-high dominated; first GPU wave ≈ 50-70 GPU-h total
```

Harness unchanged (Slurm only; no verification ceremonies; single-user). Copy the two
input docs into the repo (`docs/wave2_portfolio.md`, plus the four design files under
`docs/wave2_designs/`) so later stages are self-contained.

---

# 2. FIRST ACTION — the coordinated sealed-spend ledger

Write `docs/sealed_spend_ledger.md` and commit BEFORE anything else:

```text
SEALED-SPEND LEDGER (spend-once assets; assignment requires a frozen pipeline AND a
preregistered numerical prediction computed from support-side data only):
  MobileBCI sealed-8   : OWNED BY M35 (C-1). Not delayed by anyone. DT-Gibbs G2 arms may
                         join that single opening ONLY if G2's pipeline freezes before
                         M35's sealed shot runs; otherwise DT-Gibbs confirms on SHU
                         Day-4/5 later. One opening serves all riders.
  PhysioMotion-10      : RESERVED for MOKA M-F (only if M-C fires).
  BrainID Day-200      : RESERVED for the cross-day drift falsification (P3-class test),
                         spent inside whichever paper needs it, prediction attached first.
  SHU Day-4/5          : DT-Gibbs alternate / OPERA Regime-B panel. Never double-spend.
No sealed byte is read in WAVE2-T1.
```

# 3. Preregistration (`reports/wave2_preregistration.md`, commit before submissions)

Freeze ALL of the following now (Tier-1 AND Tier-2 gates; no changes after commit):

```text
LANGUAGE RULE (from the hostile audit): the delivered/ceiling conversion band is 0.3-0.5
  (residual semantics unified: state per-leg whether the oracle number is additive
  residual or total, and compute both readings); D ∈ [τ², 2.4τ²] with D≈τ² labeled the
  optimistic endpoint. The AUC 0.70-0.80 retrodiction is a consistency check (contains an
  unstated low-dimensionality assumption), not a confirmation.
NOT RUN (superseded/barred): the P10 coherence-guard experiment (self-contradicts
  manifold-displacement principle — DT-Gibbs likelihood routing replaces it); the
  importance-reweighting prior probe (C4 letter).

SHARED LAYER outputs (Tier-1, CPU): per-panel {τ², W, D, Σ_drift} with the unified
  semantics; the P2 ledger test = does the measured (W, D) reproduce λ̂≈0.88-0.92 and the
  0.3-0.5 conversion band jointly (report PASS/SOFT-FAIL; SOFT-FAIL downgrades Σ_drift to
  an empirical object everywhere and strips ρ̂-prediction language from sealed plans).
P1 SAME-BLOCK REPLAY (banked checkpoints, inference): support and query operators drawn
  from the SAME chronological block (drift removed) vs the standard split — if the
  conversion fraction rises toward λ̂ (~0.9), drift is the binding loss and OPERA-E2's
  transport customer evaporates; if it stays ~0.3-0.5, prior/interface quality binds and
  DT-Gibbs's prize shrinks. This adjudication GATES Tier-2 scale decisions.
P7 ANCHOR DOSE-RESPONSE (banked): interpolate a0 between correct and wrong operators;
  prediction = harm grows superlinearly with misalignment (manifold displacement).
DRIFT-WIDENED W4 REPLAY (banked V44-S1 checkpoints): K-chain draws from
  N(C̄_s, Σ_post ⊕ Σ_drift), no Gibbs; endpoints = coverage bands vs W4's 0.271/0.440/0.516
  and CRPS vs 0.1529 — the UQ band-repair row decoupled from Gibbs risk.

MOKA M-A (CPU; STEP ZERO = verify on disk that the frozen v4/v5 MobileBCI motion blocks
  + synchronized IMU exist and are readable, AND pull the v4/v5 closure wording from the
  repo branches to confirm it covers temporal-support conditioning only): artifact-band
  IMU-predictable energy fraction per subject/protocol. GO to M-B iff median fraction
  ≥ 0.10. M-B (CPU): own-vs-population motion-operator headroom (V19-style paired
  construction on motion blocks). GO iff mean ≥ +0.02 with CI-low > 0 in ≥1 protocol.
OPERA A0 (CPU): corpus/corruption audit with the e-EXOGENEITY BOUND AS A HARD GATE
  (preregistered bound on neural leakage into the EOG reference; Eye-BCI optical
  reference noted as the only clean escape). A1 (15-25 GPU-h, first GPU wave): the
  synthetic double-dissociation — clean-window-selected prior shows the U° energy
  deficit and over-subtraction; ambient/EM-trained prior does not. GO iff the
  dissociation is clean in BOTH directions (its failure falsifies the M13R causal
  diagnosis and closes OPERA with a publishable diagnostic note).
DT-GIBBS G0 (CPU): drift-prior construction from the shared layer + prior-predictive
  coverage of held-out own-query operators (80% nominal within [0.70, 0.90]).
  G1 (5-10 GPU-h, first GPU wave): leakage/attribution sanity on frozen V44-S1
  checkpoints with TWO-SIDED equivalence-bound gates sized against the GB-1 GO margin
  (+0.03): clean-pass bias and C_true-bias must be bounded, not merely n.s.; the
  predicted C-step bias direction is preregistered as TWO-SIDED (impure prior may
  inflate Ĉ → over-subtraction). NO-GO closes the semi-blind family with the diagnosis.
THRESH T0 (CPU): atlas-calibrated task priors, measured d_eff, closed-form reference
  curves UNDER THE MEASURED ANISOTROPY (if the anisotropic derivation kills the harm
  prediction on paper, the demo shrinks to transition-only before any GPU), full T1
  preregistration. T1a (10-12 GPU-h, first GPU wave): operator-space ICL grid; the
  transition metric and falsification branches per the design file.
TIER-2 GATES (frozen now, run later only on GO): DT-Gibbs G2 (40-80 GPU-h; MUST
  adjudicate before M35's sealed shot or reroute per the ledger); THRESH T1b (40-50;
  only if T1a fires AND the standalone paper is wanted); OPERA A2 (60-120; behind A1);
  MOKA M-C (60-120; behind M-B GO + closure-wording verification). Priority if compute
  or attention binds: G2 > T1b > M-C > A2.
```

# 4. Execution order

```text
1. sealed ledger commit → preregistration commit
2. shared atlas/drift layer (CPU array; extends the M0 atlas with drift statistics)
3. in parallel (CPU): P2 ledger test; MOKA M-A step-zero + M-A + M-B; OPERA A0;
   DT-Gibbs G0; THRESH T0
4. banked-checkpoint inference batch (one small GPU array, ~5-10 GPU-h total):
   P1 same-block replay, P7 dose-response, drift-widened W4 replay
5. DECISION POINT 1 (write decisions/wave2_dp1.json): P1 attribution verdict; P2
   PASS/SOFT-FAIL; M-B GO/NO-GO; G0 coverage; A0 exogeneity gate; T0 anisotropy verdict
6. first GPU wave (auto-proceed for whichever passed its gate): DT-Gibbs G1 (5-10),
   OPERA A1 (15-25), THRESH T1a (10-12) — any GPU partition, --time=04:00:00 arrays
   (A1 may need 24 h cells on A100/H100)
7. DECISION POINT 2 (decisions/wave2_dp2.json): G1 two-sided leakage verdict; A1
   double-dissociation verdict; T1a transition verdict
8. aggregate, write reports/wave2_t1_report.md, ledger section, commit, push, STOP
```

Stop after Decision Point 2. Report verbatim: the shared-layer {τ², W, D} table per
panel, P1/P2/P7/W4-replay results, M-A/M-B numbers, A0 exogeneity bound, G0 coverage,
T0 anisotropy verdict, and the three first-wave GPU verdicts. Tier-2 launches are the
operator's call (G2's sealed-shot timing especially).

# 5. Kickoff prompt

```text
Read WAVE2_T1_Shared_Layer_And_Cheap_Kills_Server_Instructions.md in full and execute
it. Create branch codex/wave2-t1 (base = current flagship tip). Commit the sealed-spend
ledger FIRST, then the preregistration (all Tier-1 and Tier-2 gates frozen). Build the
shared atlas/drift layer; run the CPU battery (P2, MOKA M-A/M-B with its step-zero data
verification, OPERA A0, DT-Gibbs G0, THRESH T0) and the banked-checkpoint inference
batch (P1, P7, drift-widened W4 replay); write Decision Point 1; auto-proceed to the
first GPU wave (G1, A1, T1a) for whichever designs passed; write Decision Point 2 and
the report; commit, push, stop. Slurm only; no sealed reads; no verification
ceremonies; the P10 coherence-guard and importance-reweighting probes are barred.
```
