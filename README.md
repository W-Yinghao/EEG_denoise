# EEG_denoise — subject-calibrated ocular-artifact removal for mobile EEG

One branch, one tree, every record. `consolidated` carries the manuscript branch
(`codex/paper-final-runs`) plus the complete experimental lineage of the project
(69 branches, June–September 2026), vendored with provenance so that nothing
requires another branch or worktree. Start here; `BRANCHES.md` says what each
branch was and found, `PROVENANCE.md` says exactly how this tree was assembled.

## What the project is

A diffusion-based EEG denoiser whose guide is a **participant-calibrated
propagation operator**: a 46×2 map from the recorded bipolar EOG to the scalp
and ear channels, estimated from a 120-s calibration segment, shrunk toward a
population operator by an empirical-Bayes reliability gate, and fed as the
guide of a shared conditional diffusion model (the frozen V44-S1 system). The
paper evaluates it on the MobileBCI corpus (15 development + 8 sealed
participants; standing / slow walking / fast walking; ERP and SSVEP tasks) with
paired restoration error, natural-recording endpoints, predictive intervals,
downstream decoding, a sealed EEGEyeNet confirmation, and head-to-head
reproductions of EEGDfus, D4PM and DS-DDPM. The numbers are written up in
`docs/results/RESULTS_PAPER_FINAL.md` and `docs/results/WAVE6_RESULTS.md`; the
draft is `paper/current/`.

## Layout

| path | what it is |
|---|---|
| `src/eeg_scad/` | the frozen V44-S1 system (data registries, EB gate, guided diffusion, evaluation) — exactly `codex/rgcc-eog-v44` |
| `src/eeg_chart/` | flagship M0/M13/M35 + wave2–4 machinery (sealed confirmation, posterior sampling) |
| `src/eeg_cgdr/`, `src/eeg_cspd/` | the benchmark/exploration harness behind `results/cgdr/` (D4PM, EEGDfus, Klados, SGEYESUB, BCI2a/2b lines) |
| `saddpm/` | the original SADDPM re-implementation (legacy; its scripts are in `scripts/legacy_saddpm/`, its README in `docs/legacy/`) |
| `scripts/paper_final/` | every paper-final runner and analyzer (T1–T6, D-wave, BCI-IV-2a, wave-6 x1–x6, reproductions), the figure library (`figures/`), the SGEYESUB-style exports (`sg_export/`) |
| `scripts/iris/` | the sealed-55 EEGEyeNet confirmation machinery |
| `scripts/slurm/` | SLURM launchers (always `--time=23:59:59`, `--exclude=node54`) |
| `configs/` | every experiment's YAML (data roots, folds, training) |
| `results/` | banked outputs of every experiment — index in `results/README.md` |
| `reports/` | pre-registrations, stage reports, job ledgers — `reports/README.md` |
| `paper/` | manuscript drafts, TAAS revision, figure library, references — `paper/README.md` |
| `docs/` | plans, ledgers, server instructions, results write-ups — `docs/README.md` |
| `paper_final_arrays/`, `results/paper_final/paper_final_arrays/` | the small arrays the figures are drawn from (tracked on purpose) |
| `runs/` | per-job SLURM run records (audit trail) |
| `figures/`, `splits/`, `decisions/`, `third_party/` | lineage-era artefacts referenced by `src/eeg_chart` and the wave reports |
| `archive/lineage_src/` | lineage versions of the files where the paper branch's version was kept (PROVENANCE §3) |
| `provenance/` | machine-readable manifests behind PROVENANCE.md |
| `tests/` | 1,236 tests; `pytest.ini` puts `src` on the path |

## Running things

- Environments: `icml` (Python 3.9, torch 2.8 cu128; GPU work, mne, asrpy) and
  `eeg2025` (Python 3.13, torch 2.6; pytest, the EEGDfus upstream). The
  interpreter paths are spelled out in `scripts/slurm/*.sbatch`.
- Everything that computes runs through SLURM (`sbatch scripts/slurm/<job>.sbatch`);
  the login node is for editing and `pytest`.
- `pytest tests` from the repo root. The 25 lineage provenance modules listed in
  `tests/LINEAGE_PROVENANCE_TESTS.txt` are skipped unless
  `DENOISENET_LINEAGE_TESTS=1`; four paper tests need the upstream checkouts in
  `.external/` (populated by `scripts/slurm/jobs/benchmark_source_checkout.sbatch`).
- Data live under `/projects/EEG-foundation-model/` (never in git); the V44-S1
  checkpoints under `/projects/EEG-foundation-model/derived/denoiseNet/rgcc_eog_v44/`.
  `DENOISENET_V44_SRC`, `DENOISENET_V44_RESULTS`, `DENOISENET_V43_STATE` and
  `DENOISENET_FLAGSHIP_ROOT` re-point the code at the original worktrees if a
  bit-for-bit replay against them is ever needed.
- Figures: `python scripts/paper_final/figures/fig_<name>.py` writes PDF+PNG to
  `paper/figures/`; the SGEYESUB-style set lives in `scripts/paper_final/sg_export/`.

## Discipline this repo follows

Pre-registration frozen and committed before compute; a single-unit probe with
QC gates before any fleet; results committed separately from interpretation;
defects disclosed and preserved rather than silently repaired; negative results
reported in full. The manuscript is written by the project owner; the code and
records here are the material.

## History

`master` (the June SADDPM re-implementation) → `codex/paper-final-runs` (the
manuscript branch, August–September) → `consolidated` (this branch). Every
`codex/*` experiment branch is an ancestor of `consolidated` through an explicit
`-s ours` merge, so `git log` reaches all of it and `git branch --merged
consolidated` lists all 69. The other branches and their worktrees are frozen
records; nothing here needs them.
