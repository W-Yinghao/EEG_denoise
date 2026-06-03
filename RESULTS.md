# RESULTS — logged numbers & assumptions actually used

This file is the running record of what was actually run and the §12 **Assumptions Ledger**
updated with the values used (per handoff working-rule 4). Updated at each milestone.

## Server / environment (recorded at M0)

| Item | Value used |
|------|------------|
| Host (login) | `nodecpu11` (no GPU) |
| GPU execution | Slurm partition `V100` (V100-16GB / V100-32GB nodes) |
| Conda env | `eeg2025`, Python **3.13.7** |
| Dependency change | `pip install moabb` → added moabb 1.5.0 + 8 deps; **only** `pymatreader` bumped 1.1.0→1.2.3; torch/numpy/mne/scipy/sklearn/pandas/matplotlib unchanged |
| torch | 2.6.0+cu124 (CUDA build 12.4) |
| mne / moabb | 1.11.0 / 1.5.0 |
| numpy / scipy / scikit-learn / pandas | 2.4.4 / 1.17.0 / 1.8.0 / 3.0.1 |
| einops / pyyaml / tqdm / matplotlib / wandb | 0.8.2 / 6.0.3 / 4.67.1 / 3.10.8 / 0.24.2 |
| Data source | **Local, pre-staged**: `/projects/EEG-foundation-model/BCI-IV` (A0{1..9}{T,E}.mat, md5-identical to MOABB's BNCI2014_001 source). Symlinked into MOABB's cache by `scripts/link_local_dataset.py` → **no re-download** (verified: loading A02 triggers 0 downloads). |
| Global seed | 42 |
| W&B | authenticated via `~/.netrc` (`api.wandb.ai`). Project default `saddpm` (configurable); **entity/project to confirm before first training run (M2)**. |

## Assumptions ledger (§12) — value used

| Item | Source | Value used |
|------|--------|------------|
| Dataset, channels, sessions, classes, sample rate | paper | BCI-IV-2a (`BNCI2014_001`), 22 EEG (+3 EOG), T/E, 4-class, 250 Hz |
| Band-pass / notch / window / z-score | paper | 1–50 Hz FIR (firwin), 50 Hz notch, 2 s/0.5 s windows, per-channel z-score |
| Diffusion T, β type, objective | paper | T=1000, linear β **used + numerically verified (M1)**; ε-prediction (`L_simple`) — *(M2)* |
| Subject embedding dim, emb. weight decay | paper | 128, 1e-4 — *(M3+)* |
| Backbone family, FiLM, time emb, attention, dual decoder, 3 losses | paper | 1D-Conv U-Net; FiLM; sinusoidal; self-attn; content+individual+classifier — *(M2–M4)* |
| Optimizer/lr/schedule/batch/epochs | paper | Adam, 1e-4, cosine, 64, 100 — *(M2+)* |
| `[DD-1]` reverse noise = `ε_θ+ε_φ` | assumed | sum — *(M4)* |
| `[DD-2]` denoising scheme | assumed | SDEdit, `t*`=200 — *(M5)* |
| `[DD-3]` `x₀` / clean target | assumed | preprocessed EEG as `x₀` (Phase 1) |
| `[DD-4]` test-time embedding for unseen subject | assumed | source embedding (Phase 1) — *(M5+)* |
| `[DD-5]` downstream classifier | assumed | EEGNet-8,2 — *(M6)* |
| `[DD-6]` U-Net length | **used** | pad 500→**512**, symmetric zero-pad (6,6), recorded for un-pad |
| `[DD-7]` attention placement | assumed | bottleneck (len 64) — *(M2)* |
| `[DD-8]` `Z^c,Z^s` features | assumed | GAP of decoder penultimate (dim 128), column-mean-centered — *(M4)* |
| `[DD-9]` ArcFace m, κ | assumed | 0.5, 30 — *(M4)* |
| `[DD-10]` loss weights | assumed | λ_r=1, λ_o=0.1, λ_a=0.1 — *(M4)* |
| `[DD-11]` subject-correlation metric | assumed | Pearson on trial-mean descriptor — *(M7)* |
| β range | assumed | **used** 1e-4→0.02 (linear) |
| U-Net widths/blocks/norm/dropout | assumed | 64×(1,2,4), 2 blocks, GN(8), 0.1 — *(M2)* |

*(M…)* = value is declared but first exercised at that milestone.

### Additional choices made at M0 (logged, not silent)

- **MI epoch interval [DD, this build]:** trials epoched at `tmin=2.0 s`, `tmax=6.0 s` relative to
  the class-cue event — the standard 4 s motor-imagery interval for BCI-IV-2a (matches MOABB's
  default `interval`). Configurable in `configs/data.yaml` (`epoch.tmin_s/tmax_s`).
- **Windows per trial:** epoch length 1001 samples → sliding 500/125 → starts {0,125,250,375,500}
  = **5 windows/trial**.
- **Trials per session:** **282** valid trials (not 288): MOABB drops ~6 artifact-/NaN-marked
  trials per session. → **1410 windows/session** for A01 (both T and E).

## M0 — env + data ✓

- `scripts/check_env.py`: all required packages present; CUDA correctly reported unavailable on
  the login node (use `--require-cuda` inside Slurm GPU jobs).
- `scripts/m0_load_subject.py --subject 1`:
  - A01 `0train` (T): windows **(1410, 22, 512)**, labels {left 340, right 360, feet 350, tongue 360}.
  - A01 `1test` (E): windows **(1410, 22, 512)**, labels {left 345, right 360, feet 350, tongue 355}.
  - z-score check: per-window mean ≈ 0, overall std ≈ **0.987 = √(500/512)** (unit-variance over the
    500 real samples diluted by the 12 zero-pad samples) — confirms z-score + pad are correct.
  - figure: `artifacts/figures/m0_subject01_window0.png` (22 z-scored EEG channels, 1–50 Hz, EEG-like).
- `tests/`: 8 unit tests pass (windowing shapes/content, z-score stats, pad/unpad round-trip, config).

## M1 — diffusion core ✓

- `saddpm/diffusion`: `DiffusionConfig` (`configs/diffusion.yaml`: T=1000, linear β∈[1e-4, 0.02]),
  `GaussianDiffusion` with precomputed buffers (β, α, ᾱ, √ᾱ, √(1-ᾱ), √α, √β) and `q_sample`
  (one-shot reparameterization) + `q_sample_stepwise` (iterated single steps).
- **Numerical forward-marginal check (§1.3) PASSES:** for t ∈ {0,9,99,299,499,799,999}, both the
  iterated single-step path and the one-shot reparameterization match the closed form
  `N(√ᾱ_t x₀, (1-ᾱ_t) I)`. Worst error over all metrics (mean, variance, off-diagonal covariance)
  = **0.0267 < 0.05** tolerance (n=20000 Monte-Carlo). Errors grow with t purely from MC variance;
  off-diagonal covariance ≈ 0 confirms the isotropic `(1-ᾱ_t)I` structure. Schedule sanity:
  ᾱ₀=0.99990, ᾱ_{T-1}=0.00004.
- figure: `artifacts/figures/m1_marginal_check.png` (schedule curves + error-vs-tolerance).
- `tests/`: **13 total unit tests pass** (+5 diffusion: schedule endpoints/monotonicity, q_sample
  zero-noise + shape, marginal-match).
