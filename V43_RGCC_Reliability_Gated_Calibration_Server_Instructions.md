# V43 — RGCC
## Reliability-Gated Calibration Conditioning for Subject-Aware Diffusion EEG Denoising
### Server execution instructions (Claude Code)

V43 does not chase MATCH>POP as its headline. The V42R ORACLE arm (the 120-s query-fitted
transfer, i.e. the perfect version of any support estimator) already bounds that gain at
+0.006267 with CI upper +0.019680. The headline is the **floor**:

> A fixed closed-form empirical-Bayes gate on the calibration-estimated transfer state makes
> subject-aware conditioning safe: wrong-donor harm (+0.0515) and short-support harm
> (+0.0822 at 10 s) are eliminated, support-duration behavior becomes monotone, and the
> gated route is non-inferior to the population route within a preregistered margin.

A separate oracle-trained ceiling probe (S1.5) is the only legitimate way to reopen the gain
claim. Its GO/NO-GO rule is frozen below before anything runs.

---

# 1. Workspace and Git

Base:

```text
branch:
codex/cleanroom-calib-saddpm-cond-v42r

commit:
ec635b2177442cd54620b8075366c81d42d5704a
```

Create:

```text
worktree:
/home/infres/yinwang/denoiseNet_rgcc_v43

branch:
codex/rgcc-v43
```

At startup:

```bash
git fetch --all --prune
git rev-parse codex/cleanroom-calib-saddpm-cond-v42r
```

Commit directly to `codex/rgcc-v43` after each stage. Push at will. No PR, no master merge.

---

# 2. Harness (lightweight — single-user project)

This project has exactly one human operator. Apply the MINIMAL harness:

KEEP (scientific, non-negotiable):

```text
all compute via sbatch (login node = editing, submission, tiny aggregation only)
sealed MobileBCI participants (the 8 sealed IDs) never read
query EOG / query-fitted transfer used ONLY in evaluator and ORACLE roles, never in a deployable arm
V42R artifacts and results read-only (no file under results/calib_saddpm_cond_v42r or the
  job_941770 checkpoint tree may be modified)
preregistration decisions in section 5 are frozen BEFORE the first submission and may not be
  tuned to outcomes afterwards
common random numbers across arms (reuse the existing sample_bank seed convention verbatim)
```

DROP (do not build, do not run):

```text
no sha256 checkpoint-binding ceremonies
no clean-archive / clean-checkout test replays
no dual-implementation replay comparisons
no per-file hash ledgers, CAS, provenance freezes, attachment audits
no gate-status JSON bureaucracy beyond the single decision JSON per stage
```

Ledger: append at most one short section per completed stage to
`docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md` (a few lines: decision + headline
numbers). Nothing else.

Unit tests: a handful of pytest tests for NEW code only (bypass identity, shapes,
normalization reuse). No test-suite ceremonies.

Slurm partitions available on this cluster:

```text
CPU:  CPU, cpu-high
GPU:  H100, A100, L40S, A40, V100
```

Choose by availability (`sinfo`): replay/inference jobs run fine on any GPU partition
(prefer V100/A40/L40S to keep A100/H100 free); the S1.5 training cells prefer A100 or H100.

Environments: reuse exactly what the V42R sbatch scripts use (do not create or modify conda
environments).

---

# 3. Frozen inputs

```text
checkpoints:
/projects/EEG-foundation-model/derived/denoiseNet/calib_saddpm_cond_v42r/job_941770/
  fold_*_seed_*/best.pt        (glob the actual fold/seed directories; expect 10 cells)

prepared records (MobileBCI panel):
/projects/EEG-foundation-model/derived/denoiseNet/counterfactual_operator_headroom_v19/prepared/

configs:
configs/calib_saddpm_cond_v42r/data.yaml   (support_seconds 30, source_root as above)
configs/setcalibdiff_v25/{data,folds}.yaml (folds, qgen/qnatural constants)

code to reuse unmodified:
src/eeg_scad/models/calib_saddpm_cond_v42r.py   (model; transfer state consumed at every DDIM step)
src/eeg_scad/data/artifact_transfer_v41r.py      (TransferRegistry, ridge fit, episode sampler)
src/eeg_scad/training/train_v42r.py              (sample_bank, evaluate_joint, paired metrics)
src/eeg_scad/evaluation/aggregate_v42r.py        (participant-first aggregation pattern)
scripts/slurm/v42r_natural.sbatch, v42r_train.sbatch (templates to clone)
```

Semantics you must not get wrong (they were mislabeled once already):

```text
POP arm               = real population conditioning state
                        (mean of fold-TRAIN owners' 30-s prefix transfers, normalized like MATCH)
NO_TRANSFER_BRANCH    = the exact bypass (transfer_enabled=False zeroes Delta_transfer)
ORACLE arm            = ridge transfer fit on the Qgen interval (samples 15000..27000), evaluator-only
support region S120   = samples 0..12000 (0-120 s); guards 120-150 s and 270-300 s;
                        natural queries start at sample 30000
common-noise seed     = 420000 + fold*100 + (seed % 100), inside sample_bank
```

Frozen V42R reference numbers (paired panel, participant-first temporal RRMSE):

```text
RAW 0.714933   POP 0.632308   (+1.198670 dB, q99 0.989213)
MATCH-POP              -0.0000268   CI [-0.005375, +0.005965]
MATCH-WRONG            +0.051454    (=> WRONG-POP harm ~ +0.0515)
MATCH-NO_TRANSFER      +0.089110
ORACLE-MATCH           +0.006267    CI [-0.001607, +0.019680]
support duration 0/10/30 s: 0.637565 / 0.719760 / 0.638260   (10-s spike = +0.0822 vs 0 s)
```

---

# 4. S0 — panel note, preregistration, EB state builder

## 4.1 Panel provenance note (5 minutes, no job)

Read `configs/cgdr/counterfactual_operator_headroom_v19.yaml` and confirm
`data_root: /projects/EEG-foundation-model/mobile_bci`. Record one sentence in
`reports/v43_stage0.md`: the 15-participant 46-EEG + 4-eye-electrode panel is MobileBCI
(OSF R7S9B), 16 development / 8 sealed. This is a fact note for the manuscript, not an audit.

## 4.2 Preregistration (write BEFORE any submission)

Write `reports/v43_preregistration.md` containing verbatim:

```text
V43 preregistration — frozen before first Slurm submission.

GATE RULE (the method):
  lambda = clip(tau2 / (tau2 + within/4), 0, 1)          [V19 closed form, scalar, primary]
  tau2   = mean squared deviation of fold-TRAIN owners' 120-s transfers around the
           population transfer of the (session, task) cell
  within = mean squared deviation of the four 30-s sub-block transfers around the
           full 120-s transfer of the support owner
  h_gated = h_pop + lambda * (h_120 - h_pop)
  HARD GATE: effective support < 60 s OR within above the frozen threshold
             (95th percentile of fold-train within values) -> lambda := 0 exactly.
  lambda = 0 MUST produce a signature array bit-identical to the frozen POP state
  (quality features clamped to the fold-train 30-s range; registry30 continuous
  center/scale reused for normalization so POP comparability is exact).
  Secondaries (reported, never primary): per-row lambda; raw un-shrunk 120-s state.

S1 (frozen-checkpoint floor probe) ADJUDICATES ONLY:
  F1 wrong-donor harm elimination:
     mean[RRMSE(WRONG_EB120) - RRMSE(POP)] <= +0.010
     AND reduction vs frozen WRONG harm has bootstrap CI-low > 0.
  F2 short-support harm elimination:
     mean[RRMSE(MATCH_EB10) - RRMSE(POP)] <= +0.010   (frozen 10-s spike is +0.0822).
  F3 provisional non-inferiority (frozen checkpoint, OOD-caveated):
     mean[RRMSE(MATCH_EB120) - RRMSE(POP)] <= +0.005.
  The MATCH_EB120 - POP gain reading is NON-ADJUDICATING in S1 (the checkpoint was
  trained on 30-s states; a positive or null here neither opens nor closes the gain claim).
  Definitive non-inferiority margin for the retrained model (S2, if run): delta = 0.002.

S1.5 (oracle-trained ceiling probe) GO RULE for the gain leg:
  train with query-fitted (generative-truth) conditioning; evaluate ORACLE vs POP
  on held-out participants of the same cells.
  GO  iff mean[RRMSE(POP) - RRMSE(ORACLE)] >= +0.020 AND bootstrap CI-low > +0.005.
  NO-GO -> the waveform-level gain claim is declared dead on this panel; V43 proceeds
  floor-only; S2 trains the gated model for the floor endpoints only.

STATISTICS: participant-first (n=15) mean, median, positive count, 5000-draw bootstrap;
Holm over the family {F1, F2, F3}. No other corrections, no post-hoc endpoint additions.
```

## 4.3 EB state builder (new code, ~250 lines)

`src/eeg_scad/data/eb_transfer_v43.py`:

```text
class EBTransferRegistry(seconds=120 | 10):
  - per (owner, session, task) cell: fit the full-prefix 46x2 ridge transfer in V41R
    coordinates (same ridge ratio, prefix-only robust EOG center/scale, fold eeg_scale),
    plus four equal sub-block transfers (30 s each for seconds=120; 2.5 s each for seconds=10)
  - population transfer/quality: reuse registry30's values unchanged (do NOT refit)
  - lambda per the preregistered rule; emit h_gated
  - signature = concat(((transfer, log-rownorm, quality) - registry30.continuous_center)
                        / registry30.continuous_scale, eye(46))
    with quality features clamped to the fold-train 30-s min/max
  - write eb_state_manifest.csv per fold: owner, lambda, tau2, within, hard-gate flag
  - assert: lambda==0 -> signature array-equal to registry30's POP signature (pytest)
```

`src/eeg_scad/cli/run_v43.py` with subcommands `stage1`, `stage15`, `aggregate`.

---

# 5. S1 — frozen-checkpoint floor probe

Per fold-seed cell (10 cells, Slurm array):

1. Load `best.pt` (EMA weights, as `paired_channel_evaluator` does).
2. Rebuild the identical test bank: `TransferEpisodeSampler(data, fold, "test", seed+3, registry30).sample_balanced(8)`.
3. Evaluate arms with `sample_bank(seed = 420000 + fold*100 + seed%100, transfer_enabled=True)`:

```text
in-run anchors : POP, MATCH            (must land on the frozen numbers; they share the noise)
new arms       : MATCH_EB120           (gated 120-s state, primary)
                 MATCH_RAW120          (lambda forced to 1, secondary)
                 MATCH_EB10            (gated 10-s state -> hard gate fires -> expect ~POP)
                 WRONG_EB120           (each wrong donor's 120-s state gated with ITS OWN lambda)
optional       : MATCH_EB120_PERROW
```

4. Write `results/rgcc_v43/stage1/fold_F_seed_S/stage1_result.json` with per-episode paired
   metrics (reuse the existing paired-metric code).

Aggregate (CPU job): participant-first contrasts per section 4.2, decision JSON
`results/rgcc_v43/stage1/decision.json` with `{F1, F2, F3, gain_reading_nonadjudicating}`,
short report `reports/v43_stage1.md`.

Slurm:

```text
state build : partition CPU (or cpu-high), ~minutes
replay      : clone v42r_natural.sbatch -> v43_stage1.sbatch
              partition V100 / A40 / L40S (any), gpu:1, --time=04:00:00, array 0-9
aggregate   : partition CPU, --time=00:30:00
```

Expected total: well under 10 GPU-hours.

---

# 6. S1.5 — oracle-trained ceiling probe

Purpose: measure the TRAINABLE ceiling of the conditioning channel. Non-deployable by
construction; label it as such everywhere.

```text
cells       : fold 0 and fold 2, seed 20261201 (2 cells)
recipe      : clone the V42R training recipe exactly (80k updates, AdamW 1e-4, EMA 0.999,
              20% population-context dropout, checkpoint by validation POP RRMSE)
change      : the conditioning state during TRAINING is the query-fitted (Qgen) transfer
              signature of each training episode's owner cell (generative truth),
              normalized with registry30 center/scale
evaluate    : ORACLE vs POP arms on the held-out test participants, common noise,
              same episode bank as S1
decision    : apply the frozen GO rule (section 4.2); write
              results/rgcc_v43/stage15/decision.json and reports/v43_stage15.md
```

Slurm:

```text
clone v42r_train.sbatch -> v43_stage15.sbatch
partition A100 or H100 (preferred; V100 fallback), gpu:1, --time=36:00:00, array of 2
```

Run S1 and S1.5 concurrently if capacity allows — S1.5 does not depend on S1.

---

# 7. S2 — gated retraining (CONDITIONAL, do not start unprompted)

Do NOT start S2 until the human operator has read the S1 + S1.5 decisions. Outline for
planning only:

```text
if S1.5 GO   : 3-seed retrain with support-duration randomization (10/30/60/120 s),
               gated states in training, reliability features under the clamp contract;
               primary endpoint MATCH_EB120 - POP with CI-low > 0 for a gain claim;
               matched DET and LINEAR (per-subject ridge subtraction + EB gate) arms included.
if S1.5 NO-GO: same retrain but endpoints are the floor set only
               (definitive non-inferiority delta = 0.002, F1/F2 on the retrained model,
               duration monotonicity 0->120 s).
either way   : ~15 cells x <=36 h on A100/H100/V100.
```

Later gates (S3: natural-route repair targeting the frozen validity gate POP remaining<1 &
attenuation>0 via training severity augmentation; BCI2b MI-kappa endpoint; lambda-privacy
curve; sealed confirmatory run) are planned in
`Downloads/EEG_denoise_TAAS_positive_method_plan.md` and will be issued as separate
instructions.

---

# 8. Prohibitions

```text
no sealed-participant reads
no natural-route training or tuning in S1/S1.5
no modification of frozen V42R results, checkpoints, or configs
no manuscript edits (taas_submission/** untouched)
no new datasets, no new model families, no representation/privacy experiments here
no tuning of any section-4.2 threshold after the first submission
```

---

# 9. Deliverables checklist

```text
reports/v43_stage0.md            panel note + preregistration pointer
reports/v43_preregistration.md   frozen decisions (section 4.2 verbatim)
src/eeg_scad/data/eb_transfer_v43.py + cli/run_v43.py + tests/unit/test_v43.py
scripts/slurm/v43_stage1.sbatch, v43_stage15.sbatch
results/rgcc_v43/stage1/{fold cells, decision.json}
results/rgcc_v43/stage15/{2 cells, decision.json}
reports/v43_stage1.md, reports/v43_stage15.md
one short ledger section (optional, a few lines)
```

Stop after S1 + S1.5 decisions are written. Report both decision JSONs and the headline
contrasts back to the operator verbatim.

---

# 10. Kickoff prompt (paste into server-side Claude Code from the repo root)

```text
Read V43_RGCC_Reliability_Gated_Calibration_Server_Instructions.md in full and execute it.
Work on a new branch codex/rgcc-v43 from codex/cleanroom-calib-saddpm-cond-v42r
(ec635b21). Single-operator project: keep the harness minimal exactly as section 2
specifies — Slurm for all compute (CPU/cpu-high for CPU jobs; H100/A100/L40S/A40/V100 for
GPU jobs by availability), no verification ceremonies, no sealed reads. Write the
preregistration file BEFORE submitting anything, then implement the EB state builder,
run Stage 1 (frozen-checkpoint floor probe, array of 10, ~4 h cells) and Stage 1.5
(oracle-trained ceiling probe, 2 training cells) concurrently, aggregate, write the two
decision JSONs and short reports, commit, and stop. Do not start Stage 2.
```
