# FLAGSHIP-M0 — Preconditions & the Channel-Ceiling Matrix
### Server execution instructions (Claude Code) — Month 0–1 of the unified flagship

This stage decides, for under ~100 GPU-hours, whether the flagship's gain thesis is alive
in ANY personalization channel, and builds the physical objects (transport geometry,
operator atlas, operator posterior) every later stage needs. Four channels, one matrix:

```text
channel            ceiling probe                       status
conditioning       ORACLE−MATCH = +0.006 [≤0.020]      BANKED (V42R; dead)
transport          ORACLE-T   (this stage, U1-a)       to measure
likelihood         ORACLE-C   (= V44-S0 subtraction)   consume result
weight-space       OSCAR Stage-A LoRA probe (U1-c)     to measure
```

KILL RULE K1 (frozen): if ALL channels' oracle ceilings land ≤ +0.020 in ALL preregistered
strata, the gain thesis is dead panel-wide; the flagship pivots to
"canonical prior at scale + ceiling matrix + certified framework + UQ" — that pivot is an
OPERATOR decision; this stage only writes the decision JSON and stops.

Companion math file: copy `PROP5_Endpoint_Correction.md` into the repo as
`docs/prop5_endpoint_correction.md` BEFORE starting — the transport implementation and its
Stage-0 gates follow it verbatim (corrected geodesic family: rotation geodesic runs from
Q̄_s, not from I; ρ=0 must be bit-identical to the POP arm; locality bound is 9 of 121).

---

# 1. Workspace and Git

```text
base branch : codex/rgcc-v43  (current tip; contains the V43 EB builder and V42R assets)
new branch  : codex/flagship-m0
worktree    : /home/infres/yinwang/denoiseNet_flagship_m0
```

V43-S2 (codex/rgcc-v43) and V44 (codex/rgcc-eog-v44) results are consumed READ-ONLY from
their branches/result trees when available; never write into their worktrees. Commit per
unit, push at will, no PR/master merge.

Harness: identical to V43 §2 — Slurm only (CPU/cpu-high; H100/A100/L40S/A40/V100 by
availability; A100/H100 at `--time=24:00:00`), no verification ceremonies, sealed cohorts
never read (MobileBCI-8, PhysioMotion-10, SHU Day-4/5, BrainID Day-200), one short ledger
section at the end.

---

# 2. Preregistration (write `reports/m0_preregistration.md` and commit BEFORE any submission)

```text
FLAGSHIP-M0 preregistration — frozen before first submission.

U0-a TRANSPORT GEOMETRY GATES (per docs/prop5_endpoint_correction.md §6, fail-closed):
  round-trip max_ρ ‖T(ρ)+T(ρ) − I‖max ≤ 1e-10 at ρ ∈ {0, 0.5, 1}
  ρ=0 bit-identity to the POP arm (transport, projector, outputs; array-equal)
  locality: Q(ρ) is the identity on W⊥ to 1e-10 (dim W ≤ 9)
  ocular-frame concentration: post-Q principal angle(Ã_s, U°) ≤ 15° for ≥ 80% of subjects
  within-subject split-half transport distance < between-subject cohort distance (median)
  κ(T_s(ρ)) reported; subjects with κ > the frozen cap flagged, not silently dropped

U0-b OPERATOR-POSTERIOR COVERAGE GATE:
  EB posterior over 46x2 operators (mean = the V43 gated operator; covariance from the
  hierarchical EB model). Held-out support-block coverage at 80% nominal must fall in
  [0.70, 0.90] per fold (participant-first).

U1 CEILING GO RULE (per channel, per preregistered stratum):
  GO iff oracle-ceiling mean ≥ +0.020 AND bootstrap CI-low > +0.005.
  Preregistered strata: {all windows} x {high-severity tercile} x {high-EOG windows},
  on each of the three panels (MobileBCI 46-ch, Klados v4 19-ch, BCI2b 3-ch).
  The conditioning channel's banked value (+0.006 [−0.0016, +0.0197], V42R) enters the
  matrix as-is and is not re-measured.

K1: if no channel GOes anywhere → gain thesis dead; write the decision and STOP.
Statistics: participant-first (or record-first for Klados), 5000-draw bootstrap,
no post-hoc strata, no threshold changes after this commit.
```

---

# 3. U0-a — transport geometry (new package, CPU + minutes)

New package `src/eeg_chart/`:

```text
transport.py   sh_lift (Perrin-style regularized SH least squares, K=121, one global
               ridge frozen by CV on training montages), cov_align (Ledoit-Wolf shrunk
               canonical covariance, AIRM Fréchet mean), ocular_procrustes (minimal
               rotation; population base rotation Q̄_s per montage), geodesic.py
               (corrected Prop-5′ family: Σ(ρ) AIRM geodesic; Q(ρ)=exp(ρ log(Q_s Q̄_s^T))Q̄_s;
               ρ_s closed-form EB with the angle cap)
atlas.py       cross-corpus operator/covariance atlas (U0-c)
analytic.py    canonical-space analytic cleaner for U1-a (below)
```

Montages for Stage 0 (in order): MobileBCI 46-ch (the V19 prepared records), Klados v4
19-ch, BCI2b 3-ch, then SGEYESUB 80-89-ch as the conditioning stress case. Run all
geometry gates; write `results/flagship_m0/u0a_geometry.json` + `reports/m0_u0a.md`.
Unit tests: the ρ=0 bit-identity and round-trip tests are mandatory pytest items.

# 4. U0-b — operator-posterior coverage (CPU)

Extend the V43 EB builder: keep the posterior COVARIANCE, not just the mean (hierarchical
EB: between-subject scatter of fold-train operators around the population operator = prior
covariance; within-subject block scatter = observation noise). Check calibration by
held-out support blocks (fit on 3 of 4 blocks, test coverage of the 4th). Write
`results/flagship_m0/u0b_coverage.json`.

# 5. U0-c — cross-corpus operator & covariance atlas (CPU array)

For every subject/record in MobileBCI (16 dev), Klados v4 (54 records), BCI2b (9),
SGEYESUB (58): fit the standard 46x2-style bipolar ridge transfer (montage-appropriate)
and the low-artifact spatial covariance; catalog between-/within-subject variances
(these are the τ² inputs for both ρ and λ EB rules), montage stats, and severity strata.
SHU/PhysioMotion: covariance only. Output `results/flagship_m0/atlas/` (CSV per dataset).
This atlas also feeds the ceiling strata definitions and later ambient-prior training.

# 6. U1-a — ORACLE-T: transport-channel ceiling (analytic, no prior training)

Measure the transport class ceiling WITHOUT training the canonical prior, using a
closed-form analytic cleaner in canonical space (Wiener/ridge with the population
covariance Σ̄ as prior + exact rank-3 least squares on the U° artifact system — implement
in `analytic.py`; this is also the future LINEAR twin). Arms (same cleaner, only the
transport varies; common windows):

```text
T-POP     T(0) = Q̄_s L_s               (population transport)
T-MATCH   T(ρ̂_s)                       (calibration-estimated, EB-shrunk)
T-ORACLE  transport estimated from the query window (evaluator-only)
T-WRONG   donor transport in recipient construction (gated by donor's ρ̂)
GAUGE-NULL random rotation on W with matched principal angles (per Prop-5′ §5.5)
```

Ceiling estimand: RRMSE(T-POP) − RRMSE(T-ORACLE) per stratum/panel. Also report
T-MATCH − T-POP (descriptive; the deployable transport effect at zero training cost).
Panels: MobileBCI paired, Klados v4 paired, BCI2b paired. CPU-heavy/GPU-trivial: run as
cpu-high array; if the SH-lift linear algebra is slow at 121-dim, a single L40S/A40 job
is permitted. Write `results/flagship_m0/u1a_transport_ceiling.json`.

# 7. U1-b — likelihood-channel ceiling (consume V44-S0) + stratified re-measurement

Consume `results/rgcc_eog_v44/stage0/decision.json` (read-only) as the likelihood-channel
ceiling row (ORACLE-C subtraction vs population subtraction) if V44-S0 has completed;
otherwise compute the identical subtraction probe here (same code path as V44-S0 spec).
Additionally re-measure the subtraction ceilings per the preregistered strata on all three
panels (CPU). Write `results/flagship_m0/u1b_likelihood_ceiling.json`.

# 8. U1-c — OSCAR Stage-A: weight-space ceiling probe (~15-20 GPU-h)

On the frozen V42R base checkpoints (read-only), per fold (5 folds, 1 seed): attach a
rank-4 zero-init LoRA to the score-net convolutions (reuse the v8 scaffold pattern
`artifact_subspace_score_lora.py` as reference, new file `src/eeg_chart/lora_probe.py`),
fine-tune ~2000 updates per held-out subject on ORACLE-operator-synthesized pairs
(x_clean_pop + C_query·e_bank — generative-truth supervision, non-deployable by
construction, labeled as such), then evaluate that subject's paired RRMSE vs the unadapted
POP route. Ceiling estimand: RRMSE(POP) − RRMSE(ORACLE-LoRA) per subject.
This is the weight-space analogue of V43-S1.5 and completes the matrix's fourth row.
Slurm: A100/H100 `--time=24:00:00`, array over folds. Write
`results/flagship_m0/u1c_weightspace_ceiling.json`.

# 9. Decision and deliverables

Assemble the CHANNEL-CEILING MATRIX (channels x strata x panels, with CIs) —
`results/flagship_m0/ceiling_matrix.csv` + figure-ready table in `reports/m0_report.md`.
Apply the frozen GO/K1 rule; write `results/flagship_m0/decision.json`:

```text
{ per-channel per-stratum GO flags, K1_fired: bool,
  u0a_geometry_passed, u0b_coverage_passed,
  transport_deployable_effect (T-MATCH − T-POP, descriptive),
  consumed: {v44_s0: ..., v43_s2: ...} }
```

Stop after the decision JSON. Report the full matrix, the geometry/coverage gate results,
and the GAUGE-NULL row verbatim. Month 1–3 instructions (prior training) follow from the
operator's read.

```text
deliverables:
reports/m0_preregistration.md, m0_u0a.md, m0_report.md
docs/prop5_endpoint_correction.md   (copied in, §5 contracts adopted)
src/eeg_chart/{transport,geodesic,atlas,analytic,lora_probe}.py + tests
results/flagship_m0/{u0a_geometry, u0b_coverage, atlas/, u1a…u1c, ceiling_matrix, decision}
one short ledger section
```

Prohibitions: no canonical-prior training in this stage; no sealed reads; no natural-route
work; no threshold changes after the prereg commit; V42R/V43/V44 artifacts read-only.

---

# 10. Kickoff prompt

```text
Copy PROP5_Endpoint_Correction.md into docs/, then read
FLAGSHIP_M0_Ceilings_Server_Instructions.md in full and execute it. Create branch
codex/flagship-m0 from the current codex/rgcc-v43 tip in worktree denoiseNet_flagship_m0.
Commit the preregistration BEFORE any submission. Build src/eeg_chart/ (transport per the
corrected Prop-5′ family — rotation geodesic anchored at Q̄_s, ρ=0 bit-identical to POP),
run U0-a geometry gates, U0-b posterior coverage, U0-c atlas (all CPU), then the three
ceiling probes U1-a (analytic transport ceiling), U1-b (subtraction ceiling; consume
V44-S0 if present), U1-c (weight-space LoRA probe on frozen V42R checkpoints, A100/H100
24 h). Assemble the channel-ceiling matrix, apply the frozen GO/K1 rule, write the
decision JSON, commit, and stop. Slurm only; CPU/cpu-high for CPU work; no verification
ceremonies; no sealed reads; no canonical-prior training in this stage.
```
