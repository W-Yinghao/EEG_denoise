# PAPER-FINAL RUNS — T1–T6 GPU inference + CPU reference rows
**For**: server-side Claude Code (Fable 5), repo `EEG_denoise`.
**Date**: 2026-08-29. **Mode**: light harness — this is paper-finishing science, not a certification campaign.

---

## Mission context

The paper (v2 draft, ACM TAAS) is written: *The Operator Carries the Subject* — subject-calibrated EOG-guided diffusion. One method: 120-s calibration → ridge propagation matrix → EB shrinkage + reliability rule (reject → withhold guide) → guide a shared population diffusion model → K=32 operator-sampled predictive intervals. All headline numbers are already final (dev +0.1428 [15/15], held-out +0.1537 [7/8], UQ 0.802/0.853 @ CRPS 0.1503). These runs fill the remaining tables and figures and upgrade the story. **Zero training anywhere. Frozen checkpoints only.**

## Ground rules (all of them)

1. **No retraining, no temperature retuning, no touching the sealed EEGEyeNet-55 block, no re-litigating held-out point estimates** (those are final as banked).
2. Fixed seeds (reuse 20261201 conventions); participant-first aggregation; 5,000-resample participant bootstrap for CIs, as in every prior campaign.
3. **No verification ceremonies.** No sha256 rituals, no preregistration documents, no clean-checkout tests. One `RESULTS_PAPER_FINAL.md` (append per task) + stored `.npz` arrays is the entire deliverable. If something deviates from plan, write one sentence saying what and why.
4. Pipeline sanity check before new cells: reproduce dev MATCH ≈ 0.4310 and NO-guide ≈ 0.5738 on the standard dev episodes with the frozen fold models. If they reproduce, proceed; if not, fix the pipeline first.
5. Slurm: single-GPU jobs, any free partition of {H100, A100, L40S, A40, V100}; CPU tasks on {CPU, cpu-high}. Total GPU budget ≈ 6 GPU-h — everything here is inference.
6. Work on a new branch (e.g. `codex/paper-final-runs`), commit and push results + arrays at the end. You have full autonomy on implementation details; the specs below fix only what the paper needs.

---

## T1 — Held-out predictive-interval run (~1 GPU-h) — highest priority

The UQ chapter is currently development-only. Close it on the 8 held-out participants.

- Models: the same frozen seed-20261201 fold checkpoints used for the held-out point-estimate pass; same episodes/protocol.
- Freeze ONE scalar temperature from the development set before any held-out inference: same grid rule as before (0.50–6.00 step 0.05, smallest reaching 80% coverage), applied to all 15 dev cells jointly. Then apply it unchanged.
- Run K=32 stochastic trajectories with the adopted width policy (operator-posterior inflation + that temperature). Also score the temperature-only and raw-sample policies on the same trajectories for context.
- Report: coverage at 50/80/90%, Gaussian CRPS, risk–coverage area; per-participant coverage spread. Descriptive report of whatever comes out — the paper's question is simply whether 80/90 hold near nominal on unseen users.
- Store per-window mean + width arrays for at least two held-out participants (feeds T6).

## T2 — Calibration-duration curve, 30/60/90/120 s (<1 GPU-h)

Potential headline upgrade: how little calibration buys how much.

- Truncate each dev calibration prefix to 30/60/90/120 s; recompute scaling, ridge, shrinkage closed-form (at 30 s the block structure degenerates — handle it however is cleanest and say what you did).
- Two readings per duration: (a) system reading — reliability rule active (it will reject 30 s by design; that is a result, plot it as the rule firing); (b) rule-off reading — raw curve without the rejection floor, so the paper can show *why* the floor sits at 60 s.
- Evaluate matched vs unguided on the standard dev paired episodes, seed 20261201 models only.
- Output: gain-vs-duration curve with participant bootstrap CIs, both readings.

## T3 — Ablation matrix completion (<0.5 GPU-h)

The paper wants one table where a single cell lights up. Complete the 2×2 (guide ∈ {matched, none} × calibration features ∈ {matched, population}) plus one shrinkage arm:

- Missing cells to run on dev episodes (seed 20261201): **(none, population)** and **(matched, population)**.
- Extra arm: guide from the **unshrunk** ridge estimate (λ=1), matched features — quantifies what shrinkage buys.
- Existing cells (matched, matched) and (none, matched) come from stored outputs; do not re-run except via the sanity check.
- Output: the full matrix, RRMSE per cell, matched-vs-each contrasts with CIs.

## T4 — Operator lifetime / staleness curve (<1 GPU-h, mostly CPU)

The adaptation loop needs a clock: how long does a calibration live?

- CPU part: re-estimate the propagation matrix closed-form on successive 120-s windows across each dev record; plot relative RMS operator displacement vs elapsed time since the calibration prefix.
- GPU part: gain (matched − unguided) vs time-since-calibration. Prefer stratifying stored per-episode outputs by record position if position metadata exists; otherwise evaluate new paired episodes drawn from early/mid/late thirds of each record (same construction as the standard protocol). Your call.
- Output: displacement curve + gain-vs-elapsed-time curve with CIs.

## T5 — Natural-plane completion (<0.5 GPU-h)

The attenuation–retention operating-point figure needs all five conditions; mismatched and shuffled natural rows do not exist yet.

- Run **mismatched calibration** and **shuffled EOG** through the standard natural-window evaluator (same windows as the natural analysis): EOG attenuation, low-EOG retention, coherence reduction.
- Output: the 5-condition natural table (unguided / population / matched / mismatched / shuffled).

## T6 — Figure-data dump (<0.5 GPU-h)

Everything the local figure rendering needs, as `.npz` + a one-page manifest describing each array. No model access will exist locally.

1. **Waveform exemplar**: lowest-ID dev participant with complete outputs, earliest clear ocular event, 3 fixed channels (state which): EOG trace, contaminated EEG, paired reference, linear regression, unguided, matched — plus the K=32 interval band over the same window (the band should visibly widen at the blink; that is the figure's point). Also one held-out exemplar from T1 if convenient.
2. **Scalp map**: per-channel mean improvement (matched − unguided) across dev participants — one 46-vector + channel names/positions.
3. **Width-locality scatter**: per-cell propagation-width share vs the calibration within-variance v_i (from the K-chains).
4. **Operating points**: the T5 five-condition natural values; the T2 duration curve; the T4 curves; the T3 matrix — all as arrays.

## CPU reference rows (partitions CPU / cpu-high, no GPU)

Two literature anchor rows on the same protocol, both endpoints (paired RRMSE where applicable, natural attenuation/retention):

1. **ICA/ASR reference**: ASR on the standard episodes; ICA with EOG-correlation-based component removal (or ICLabel if the environment has it) on the continuous natural records. If 5.12-s episodes are too short for stable ICA, run it on continuous data only and say so — that constraint is itself reportable.
2. **Calibrated eye-subspace subtraction** (Kobler 2020 SGEYESUB style): estimate the eye subspace from the same 120-s calibration segment, project out at evaluation. Closed-form, same folds.

These are context rows, not contests; report them plainly.

---

## Deliverable

`RESULTS_PAPER_FINAL.md` on the branch with one section per task (numbers, CIs, one-line interpretation each), plus `paper_final_arrays/` with the `.npz` files and manifest. Push when done. If a task turns out to be structurally impossible with stored assets, write two sentences explaining why and move on — do not build new infrastructure to force it.
