# SADDPM — Implementation Plan

Subject-Aware Denoising Diffusion Probabilistic Model for cross-subject EEG denoising on
BCI Competition IV-2a. This plan tracks the build against
[SADDPM_IMPLEMENTATION_HANDOFF.md](SADDPM_IMPLEMENTATION_HANDOFF.md) (the authoritative spec).

Work proceeds **strictly milestone by milestone** (§11 of the handoff). Each milestone ends
with: run its sanity check / tests → commit → short status (what passed, any deviation).

## Phases

- **Phase 1 (core, faithful to paper):** SADDPM + SDEdit + EEGNet downstream + ICA baseline on
  BCI-IV-2a → 9×9 accuracy matrices + subject-correlation matrices. (Milestones M0–M7.)
- **Phase 2 (extensions):** EEGdenoiseNet paired denoising (RRMSE/CC), DL baselines, ablations.
  Built only after Phase 1 is complete and approved. (Milestone M8.)

## Milestones & exit criteria

| Milestone | Deliverable | Exit / sanity check |
|-----------|-------------|---------------------|
| **M0** env + data | `check_env.py`, moabb loader, §3 preprocessing | check_env passes; load A01; print shapes; plot one preprocessed window |
| **M1** diffusion core | `schedule.py`, `q_sample` | **Numerical forward-marginal check (§1.3) passes**; unit-tested |
| **M2** U-Net (single decoder) | `unet1d.py`, plain DDPM `L_simple` | **Overfit a single batch** (loss → ~0); samples look EEG-like |
| **M3** subject conditioning | `subject_embed.py`, `film.py` | Conditioning changes generated samples per subject |
| **M4** dual decoder + 3 losses | `dual_decoder.py`, `arcface.py`, losses | Full SADDPM trains stably; ArcFace subject acc > chance |
| **M5** SDEdit denoise | `gaussian_diffusion.sdedit` | Denoise held-out segments; sweep `t*`; visualize in/out |
| **M6** downstream + baseline | `eegnet.py`, `ica.py`, `downstream.py` | One (source,target) pair end-to-end for SADDPM and ICA |
| **M7** full sweep | `run_pairwise_matrix.py`, `subject_corr.py` | 9×9 matrices + mean/grand-mean/std; 2 correlation matrices; `RESULTS.md` |
| **M8** (optional) Phase 2 | EEGdenoiseNet, DL baselines, ablations | RRMSE/CC vs ground truth; ablation table |

## Working rules (from the handoff + user)

1. Follow every **DESIGN DECISION** default in §2/§12 exactly; never silently substitute. If a
   default is infeasible (no internet/GPU/data, missing dep), STOP and ask.
2. All hyperparameters live in YAML / dataclass configs — **no magic numbers**.
3. Seed everything; log seed. Log to **W&B and a local CSV**.
4. Maintain [RESULTS.md](RESULTS.md) with the §12 assumptions ledger updated to values actually used.
5. Numerically verify the diffusion math at **M1**; overfit a single batch at **M2** before scaling.
6. Engineering: type hints, docstrings, unit tests in `tests/`, small focused commits, no dead code.

## Server execution notes

- **Env:** conda `eeg2025` (Python 3.13.7) on this server; `moabb` added (additive install). A
  reproducible spec is in [environment.yml](environment.yml).
- **GPU:** none on the login node (`nodecpu11`). Training runs as **Slurm** jobs on partition
  `V100` (`scripts/slurm/`). `check_env.py --require-cuda` is used inside GPU jobs to fail loudly.
- **Data:** MOABB `BNCI2014_001` cache at `~/mne_data/MNE-bnci-data`. Internet is available, so
  subjects not yet cached download on first access.
