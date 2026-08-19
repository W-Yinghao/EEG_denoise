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

## Amendment S356-1 (committed before any rerun) — episode guard + subject-specificity control

**Banked state.** The frozen ITT verdict INCONCLUSIVE stands and is never edited
(gain(259) = -1.4276 [-4.4154, +0.0760]). Diagnosis, verified objectively: EVAL
subject AA4's episode bank is pathological — support injection ratio 167.2 vs
0.33-4.02 for all 14 other subjects; even BLIND reads 2.4-4.1 RRMSE on AA4 vs
0.25-0.80 elsewhere. An instrument defect in episode construction, not a
conditioning phenomenon. Beneath it: 14/15 subjects show consistent positive gains
(+0.011..+0.142) at ALL n — a pattern that, if it survives controls, bears directly
on C1 and demands the discipline below before any claim.

**Episode-validity guard (frozen).** An EVAL subject is excluded-and-counted if its
support or query injection RMS ratio falls outside **[0.1, 20]** (physically sane
range; AA4 at 167.2 is the only exclusion on current data).

**WRONG-embedding control (frozen; the missing SHUFFLED-class arm).** For COND(n) at
n = N_max and n = 30: each guarded-cohort subject is additionally evaluated under
every OTHER guarded subject's fitted oracle embedding; wrong_gain = rrmse_zero −
mean over wrong embeddings. **Subject-specificity rule: a conditioning-channel
claim requires own_gain − wrong_gain CI-low > 0.** If wrong embeddings deliver the
same benefit, the gain is PROTOCOL-GENERIC (the embedding acts as a generic
calibration knob for the injection machinery) and is reported as such — NOT as a
C1 counterexample.

**Interpretation rule (frozen, disambiguating the original overturn text).** On the
guarded cohort: (i) positive g(N_max) with positive TREND → the threshold account
resurfaces (the original overturn reading); (ii) positive g(N_max) with FLAT trend
(trend CI containing 0) AND subject-specificity passing → a scale-independent
conditioning gain on THIS panel/protocol — a scoped counterexample to C1's
universality, reported verbatim with its instrument caveats (injected episodes;
oracle embedding = support-calibration); (iii) subject-specificity failing → C1
unthreatened; the gain is calibration-protocol-generic. Robust (median) companions
reported throughout. Gates and epsilon (0.02) unchanged. Rerun = evaluate (with
saved embeddings + wrong-embedding matrix) + aggregate only; the 10 trained models
are untouched. Amended decision written BESIDE the banked one.
