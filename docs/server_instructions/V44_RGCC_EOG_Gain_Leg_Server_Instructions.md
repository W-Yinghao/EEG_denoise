# V44 — RGCC-EOG
## The gain leg: gated subject operators in the EOG-guided deployment class
### Server execution instructions (Claude Code)

V43 adjudicated the conditioning-only class: no subject-correct gain exists there (precise
null; oracle ceiling +0.006; oracle-trained route −0.051, worse than population). V44 moves
the gain question to the deployment class where the repo's strongest positive already
lives — runtime eye electrodes available, the calibrated operator applied directly:

> V19 paired subtraction: own vs population operator H_P = +0.72 [0.55, 0.89], 15/16.
> V20 natural: N_P = +0.179 (+20.3%), 15/15, randomization p = 1e-5, against a
> recipient-excluded (LOSO-style) population operator — the strong-baseline collapse
> argument does not apply at the operator level on this panel.

One architecture spans both classes:

```text
x0_hat = (y - a0) + Delta_pop + Delta_transfer
a0 = C_gated * e_bipolar(query)    EOG-guided class (V44)
a0 = 0                             conditioning-only class (reduces to V42R/V43)
```

The same EB gate (frozen in V43) produces C_gated. Query EOG is a DECLARED RUNTIME INPUT
in this class — this changes the information boundary deliberately and must be stated as
such everywhere; it is the same assumption as Gratton-style regression, ASR-with-EOG, and
the BCI2b V11.1 precedent. Mechanism honesty: any gain enters through the operator anchor
(BCI2b B1: G_A +0.0335 vs G_C +0.0011); the paper claims subject-aware SYSTEM gain, never
score-network personalization and never diffusion superiority (ledger C05).

---

# 1. Workspace and Git

```text
base branch : codex/rgcc-v43  (current tip; do not wait for S2 if it is still running)
new branch  : codex/rgcc-eog-v44
worktree    : /home/infres/yinwang/denoiseNet_rgcc_eog_v44
```

V44 is independent of V43-S2 (different model class); they may run concurrently. Never
write into the other worktree's results. Commit per stage, push at will, no PR/master merge.

Harness: identical to V43 §2 — Slurm only (CPU/cpu-high; H100/A100/L40S/A40/V100 by
availability, A100/H100 at `--time=24:00:00`), no verification ceremonies, sealed-8 never
read, frozen V42R/V43 artifacts read-only, one short ledger section at the end.

---

# 2. Preregistration (write and commit BEFORE any submission)

Write `reports/v44_preregistration.md` verbatim:

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

---

# 3. V44-S0 — subtraction probe (CPU only, run first)

Reuse the V43 EB builder (`eb_transfer_v43.py`) for C_gated / C_wrong_g in RAW OPERATOR
coordinates (the subtraction needs the physical 46x2 operator, not the normalized
signature — add a `raw_operator=True` accessor if absent). Single CPU job (partition CPU
or cpu-high):

1. Paired panel: for every test episode of every fold, compute clean_hat = y − C·e for the
   five operator arms; full-window temporal RRMSE vs clean x; also the masked (top-30%
   EOG-energy) RRMSE as a secondary row for comparability with V19.
2. Natural windows (samples >= 30000): attenuation/coherence/retention for C_gated vs C0.
3. Aggregate participant-first; write `results/rgcc_eog_v44/stage0/decision.json`
   (G0-1 GO/NO-GO + all rows) and `reports/v44_stage0.md`.

Expected: G0-1 should be large if V19/V20 generalize to full-window scoring; if it is not,
that is a decisive and cheap negative — report and stop.

---

# 4. V44-S1 — EOG-guided diffusion (training)

Model: minimal modification of `calib_saddpm_cond_v42r.py` in a NEW file
(`calib_saddpm_eog_v44.py`; V42R file untouched):

```text
inputs      : x_t, observed y, a0 (46-ch artifact estimate), timestep, transfer state
centering   : x0_hat = (y - a0) + Delta_pop(x_t, y, a0, t) + Delta_transfer(..., c)
              (a0 concatenated as 46 additional input channels to both heads)
a0-dropout  : 30% of training episodes use a0 = 0 (spans both deployment classes;
              makes the NO_A0 arm in-distribution)
everything else identical to the V42R recipe: 80k updates, AdamW 1e-4, EMA 0.999,
20% population-context dropout on the transfer state, checkpoint by validation POP RRMSE
```

Training a0 during training = C_gated of the episode owner applied to the episode's query
EOG (deployment-real); WRONG/SHUFFLED variants are inference-time interventions only.

Cells: 5 folds x seeds {20261201, 20261202, 20261203} = 15 cells.

Inference arms per cell (common noise, sample_bank convention):

```text
POP           a0 = C0 * e,        population transfer state
MATCH_gated   a0 = C_gated * e,   gated transfer state             (primary)
WRONG         a0 = C_wrong * e,   wrong transfer state (ungated)
WRONG_gated   a0 = C_wrong_g * e, gated wrong state
SHUFFLED      a0 = C_gated * shuffle_t(e)
NO_A0         a0 = 0              (conditioning-class bridge row)
ORACLE        a0 = C_query * e    (evaluator-only ceiling)
```

DET-EOG twin: same backbone, x_t := (y - a0), fixed t=0, direct MSE; 10 cells.
LINEAR-EOG rows come from S0 (no retraining).

Natural evaluation: same arms on natural windows; endpoints per G4 (attenuation, coherence
reduction, retention, PSD/covariance guardrails) — direct EOG-based endpoints, NOT the
V42R proxy-teacher remaining ratio.

Aggregate: `results/rgcc_eog_v44/stage1/decision.json` (G1/G2 verdicts, G3 rows, G4
validity flags + natural utilities), `reports/v44_stage1.md` with: the paired six-arm
table, per-participant forest data for G1, the S0 linear rows alongside, the DET twin rows,
and the NO_A0 bridge row explicitly connecting to the V43 null.

Slurm:

```text
training : v44_stage1_train.sbatch, A100/H100 --time=24:00:00 (V100 fallback 36 h), array 0-14
det twin : same template, array 0-9
inference: any GPU partition, --time=04:00:00, array 0-14
aggregate: CPU, --time=00:30:00
```

Budget expectation: ~30-60 GPU-hours.

---

# 5. Prohibitions

```text
no sealed reads
no retuning of the V43 lambda rule, hard gate, or clamp contract
no modification of V42R / V43 / V43-S2 artifacts or worktrees
no manuscript edits
no diffusion-superiority wording (C05); DET/LINEAR rows are reported wherever the
  diffusion arms are
no endpoint additions or threshold changes after the preregistration commit
```

---

# 6. Deliverables and stop points

```text
reports/v44_preregistration.md      (committed before submissions)
results/rgcc_eog_v44/stage0/{decision.json, all arm rows}   + reports/v44_stage0.md
results/rgcc_eog_v44/stage1/{decision.json, per-cell results} + reports/v44_stage1.md
src/eeg_scad/models/calib_saddpm_eog_v44.py, cli/run_v44.py, tests/unit/test_v44.py
scripts/slurm/v44_*.sbatch
one short ledger section
```

Stop points: if S0 is NO-GO, stop after S0 and report. Otherwise proceed through S1 and
stop after the S1 decision JSON; report G0/G1/G2 verdicts, the G3 positioning rows, and
the G4 natural flags verbatim. Do not start any further stage.

---

# 7. Kickoff prompt

```text
Read V44_RGCC_EOG_Gain_Leg_Server_Instructions.md in full and execute it. Create branch
codex/rgcc-eog-v44 from the current codex/rgcc-v43 tip in a new worktree
denoiseNet_rgcc_eog_v44 (independent of any running V43-S2 work). Commit the
preregistration BEFORE any submission. Run S0 (CPU subtraction probe) first; if its GO
rule passes, proceed directly to S1 (15 EOG-guided diffusion cells + 10 DET-twin cells +
inference arms + natural evaluation), aggregate, write both decision JSONs and reports,
commit, and stop. Slurm only (CPU/cpu-high; A100/H100 at --time=24:00:00, V100 fallback);
no verification ceremonies; no sealed reads; query EOG is a declared runtime input in
this deployment class only.
```
