# FLAGSHIP-M35 — Unification factorial, sealed confirmation, fresh-dataset transport
### Server execution instructions (Claude Code) — the program's final experimental round

Both gain legs are confirmed on development data (likelihood +0.143 [15/15] MobileBCI;
transport +0.016/+0.018 Holm-significant on Klados/BCI2b, zero-training, ITT). This round
(1) measures how the two physics channels COMPOSE, (2) spends the sealed card once on the
likelihood leg, (3) confirms the transport leg on a fresh dataset with a cross-day
stress, and (4) completes the deployment rows. No ownership attempts anywhere — the
reliability-not-identity limitation is documented, with abstention + certified floor as
the deployment answer. After this round the program moves to writing.

---

# 1. Workspace

```text
continue : codex/flagship-m13 (tip f9d5dc2), same worktree; branch codex/flagship-m35
compute  : CPU/cpu-high for everything except sealed/BrainID inference (any GPU, small);
           total well under 30 GPU-h
```

# 2. Preregistration (`reports/m35_preregistration.md`, commit BEFORE submissions)

```text
FLAGSHIP-M35 preregistration — frozen before first submission.

U-1 UNIFICATION FACTORIAL (Klados + BCI2b, analytic backbone, frozen R-B transport
  configs, ITT abstention):
  arms = {T-POP, T-MATCH(ρ̂)} × {A0-none, A0-POP (C0·e), A0-MATCH (C_gated·e)}, common
  windows; WRONG rows for both channels descriptive only.
  Endpoints (participant/record-first, 5000-draw bootstrap, Holm over the three tests):
  UF-1 transport gain with the anchor active: [T-POP,A0-MATCH] − [T-MATCH,A0-MATCH] > 0
  UF-2 anchor gain with transport active:    [T-MATCH,A0-POP] − [T-MATCH,A0-MATCH] > 0
  UF-3 composition: JOINT [T-MATCH,A0-MATCH] vs best single leg (report the additivity
       index (joint − base) / ((legA − base) + (legB − base)) with CI; no super-additivity
       claim — the estimand is whether the channels are complementary or redundant).

C-1 SEALED CONFIRMATION — MobileBCI sealed-8, opened ONCE (first legitimate opening in
  program history; log everything; report regardless of outcome):
  model     = the frozen V44-S1 checkpoints, 5-fold ensemble mean (seed 20261201),
              chosen and frozen HERE, before any sealed byte is read
  support   = each sealed subject's own S120 prefix (V31 contract, 120-s gated states)
  PRIMARY   = MATCH_gated − NO_A0 paired temporal-RRMSE utility on the sealed
              semi-simulated construction (identical V19-style recipe, built fresh for
              sealed subjects), mean/CI/positive count over n=8
  PRECONDITION (reported first): backbone sanity on sealed subjects — NO_A0 beats RAW
              and q99 ∈ [0.90, 1.10]; if the precondition fails, the primary is reported
              as non-adjudicable, not suppressed
  SECOND ROW = natural G4 validity (attenuation > 0, retention ≥ 0.75) + natural
              MATCH_gated − POP descriptive
  NO other endpoints. Outputs digest-frozen before the evaluator opens query EOG.

C-2 FRESH-DATASET TRANSPORT + CROSS-DAY (BrainID, Day-200 stays sealed):
  transport config (frozen): covariance alignment ONLY (G_s with the split-half
  abstention rule; Q fixed at the population base — no per-subject ocular Procrustes,
  BrainID has no dedicated EOG channels), 57-ch montage lift shared.
  support = Day-1; queries = Day-7 (primary), Day-80 (secondary/stress).
  Estimand = TG-1 (T-POP − T-MATCH > 0, analytic backbone, paired semi-sim construction
  from BrainID's own blink-rich segments); controls = WRONG donor + GAUGE-NULL; cross-day
  stability = Day-7 vs Day-80 gain ratio (descriptive). This doubles as the
  acquisition-shortcut test: a transport that survives a cap remount across 80 days is
  carrying anatomy, not acquisition.

D-1 DOWNSTREAM ROW (BCI2b): MI kappa for [T-MATCH,A0-MATCH]-cleaned vs [T-POP,A0-POP]-
  cleaned natural trials via the existing CSP-LDA/kappa harness; descriptive with CI.

P-1 TRANSPORT-STATE PRIVACY (CPU): linkage (top-1, verification AUROC) of stored
  transport states (Σ̂_s summaries + frame parameters) at ρ ∈ {0, 0.25, 0.5, 0.75, ρ̂, 1},
  Klados + BCI2b; completes the two-leg privacy story alongside the λ curve.

OWNERSHIP: no verification attempts in any leg (fourth replication documented; the
  deployment answer is abstention + the certified floor). No threshold changes after
  this commit. BrainID Day-200, PhysioMotion-10, SHU Day-4/5 remain sealed.
```

# 3. Execution order

```text
1. commit preregistration (including the frozen sealed-checkpoint choice)
2. U-1 factorial (CPU arrays) + P-1 (CPU) + D-1 (CPU) in parallel
3. C-2 BrainID: data prep (reuse the v17 prepared roles), transport build, analytic
   arms; Day-7 then Day-80
4. C-1 sealed LAST, as a single isolated job chain: build sealed paired construction,
   run frozen ensemble inference, freeze outputs, evaluate; one pass, no reruns
5. aggregate: results/flagship_m35/{u1_factorial, c1_sealed, c2_brainid, d1_kappa,
   p1_privacy}/decision.json; reports/m35_report.md; ledger; commit, push, STOP
```

Report verbatim: UF-1/2/3 with the additivity index, the sealed primary + precondition +
natural row, the BrainID TG-1/controls/cross-day ratio, the kappa row, and the ρ-privacy
table. After this the program's experimental phase is complete; the writing round is the
operator's call.

# 4. Kickoff prompt

```text
Read FLAGSHIP_M35_Unification_And_Confirmation_Server_Instructions.md in full and execute
it. Create branch codex/flagship-m35 from codex/flagship-m13 tip f9d5dc2. Commit the
preregistration first (it freezes the sealed-checkpoint choice before any sealed byte is
read). Run U-1, P-1, D-1 in parallel (CPU); then C-2 BrainID; then C-1 sealed as one
isolated single-pass job chain, outputs digest-frozen before evaluation, reported
regardless of outcome. Write all decision JSONs and the report; commit, push, stop.
Slurm only; no verification ceremonies; no ownership attempts; Day-200/PhysioMotion/SHU
sealed sets untouched.
```
