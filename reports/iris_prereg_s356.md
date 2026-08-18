# S356 preregistration — the real-data scale conditioning probe (the program's final experiment)

Committed BEFORE any execution. Funded by operator ruling 2 (digest, 2026-08-18);
after this experiment the program is declared closed. Budget ≤ 30 GPU-h.
Decision JSON → `results/iris/s356/s356_decision.json`.

## Question (C1's last standing objection)

Score-network personalization is dead three ways at 9–45 training subjects, and the
task-diversity threshold law was rejected on synthetic scales (T1a). The one surviving
objection: no REAL-data conditioning measurement exists near the threshold account's
~300-subject prediction. EEGEyeNet antisaccade is the only corpus in the program that
crosses it. Does a conditioning channel begin to pay as real training subjects scale
toward ~300?

## Cohort and acquisition (frozen)

- Antisaccade `synchronised_min` listing positions **91–370** are fetched as DEV-CLASS
  data (charter §5: they can never join the sealed block) into
  `/projects/EEG-foundation-model/eegeyenet/eegeyenet_ext/`, same tooling, header-
  verified, empties/failures counted (ITT). Sealed positions 31–90 are never touched.
- Probe cohort = dev-28 ∪ ext (expected ≈ 285 subjects total; the shortfall from the
  nominal 356 — the sealed carve-out, cross-paradigm ID unmappability barring the dots
  30, and upstream empties — is disclosed wherever the result is cited).
- **EVAL subjects (frozen, never trained on): the first 15 dev subjects in listing
  order** — AA0 AA1 AA4 AA5 AA7 AA8 AA9 AB0 AB1 AB2 AB3 AB6 AB7 AB9 AC0.
- Training pool = the remaining 13 dev + all usable ext subjects.

## Preprocessing (frozen)

Per subject: resample 500 → 100 Hz (polyphase); channels = the 8-anterior frontal
block (non-periocular, by chanlocs X) ∪ 38 evenly-spaced remaining non-periocular
E-channels by channel number → **46 channels** (the program's operator dimension);
latent reference = [VEOG, HEOG] (frozen periocular derivation; interpolated-periocular
recordings INCLUDED, status carried — the estimand is a same-data differential).
SUPPORT = first half of the recording, QUERY = second half.

## Paired episodes (V41R transfer-episode recipe, frozen)

Per subject: operator C_s = program ridge (0.05) of the 46-channel block on the
robust-centered latent, fit on SUPPORT. Episodes: x = 5.12-s windows drawn from the
lowest-30%-EOG-energy windows; e = latent from the highest-30% windows (circularly
shifted, seed 20260818); y = x + C_s e. RRMSE vs x is the metric. Train episodes come
from SUPPORT windows of training subjects; eval episodes from QUERY windows of the 15
EVAL subjects (64 per subject, frozen seed).

## Model class and arms (frozen; declared architecture-relative)

A compact DET denoiser (Conv1d encoder–decoder over 46×512, predicts the artifact;
~1M params), NEW and independent of the frozen program checkpoints — the threshold
account predicts a transition in ANY competent class, and a fresh class avoids
inheriting V44 idiosyncrasies; this choice is declared, not hidden. **No runtime
reference input** — the conditioning channel is tested in isolation (NO_A0 context),
exactly C1's estimand.

- **COND(n)**: FiLM-modulated by a learned 32-d per-training-subject embedding.
- **BLIND(n)**: the identical network with the embedding input fixed at zero
  (parameter-matched twin), trained on the same episodes/seeds.

Subject-count grid **n ∈ {30, 60, 120, 200, N_max}** (N_max = full training pool;
grid truncated from above if acquisition falls short — disclosed). Training subjects
for each n = the first n of the training pool in listing order (nested sets). Fixed
recipe both arms: 20k steps, batch 32, AdamW 2e-4, EMA 0.999, seed 20260818.

## Evaluation (frozen; the S2b oracle-embedding protocol class)

Per EVAL subject and each COND(n): the ORACLE embedding is fit on the subject's
SUPPORT episodes (Adam, 200 steps, model frozen), then evaluated on QUERY episodes;
the POP arm uses the zero embedding. Per-subject conditioning gain
g(n) = RRMSE(zero-emb) − RRMSE(oracle-emb); BLIND(n) RRMSE reported alongside
(sanity: COND(zero-emb) ≈ BLIND). Participant-first over the 15 EVAL subjects,
bootstrap 5000 seed 420.

## Gates (frozen)

- **G-S356-seal (C1 sealed)**: g(N_max) CI-high < **+0.02** AND
  [g(N_max) − g(30)] CI-high < **+0.02** (no material gain at the largest real-data
  scale ever tested, and no upward trend across a full decade of subject count).
- **G-S356-overturn**: g(N_max) CI-low > **+0.02** → the threshold account RESURFACES
  — reported immediately as a C1-overturning finding, whatever it does to the paper.
- Neither → INCONCLUSIVE, reported as such (the objection survives at reduced force).
- Guard: BLIND(n) held-out RRMSE must improve or hold with n (a broken training recipe
  invalidates the probe — instrument defect, not a verdict).

## Budget and execution

Acquisition: Slurm CPU, resumable (~65 GiB; may span multiple anonymous-quota
windows). Preprocessing: one CPU pass. Training: 10 runs (5 grid points × 2 arms),
single-GPU each; estimate 5–10 GPU-h total, cap 30. After the decision JSON and the
digest entry, the experimental program is DECLARED CLOSED per operator ruling 2.
