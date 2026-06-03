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
| Diffusion T, β type, objective | paper | T=1000, linear β **used + numerically verified (M1)**; ε-prediction (`L_simple`) **used (M2)** |
| Subject embedding dim, emb. weight decay | **used (M3)** | 128, weight decay 1e-4 on subject embeddings only (separate optimizer param group) |
| Backbone family, FiLM, time emb, attention, dual decoder, 3 losses | paper | 1D-Conv U-Net, sinusoidal time emb, bottleneck self-attn, FiLM **(M2/M3)**; **dual decoder + L_r/L_o/L_a all used (M4)**: shared subject-agnostic encoder, content (FiLM e(s)) + individual (no FiLM) decoders |
| Optimizer/lr/schedule/batch/epochs | paper | **Adam, lr 1e-4, cosine→0, batch 64, 100 epochs used (M4)**; AMP + grad-clip 1.0; M2 overfit used lr 2e-4 |
| `[DD-1]` reverse noise = `ε_θ+ε_φ` | **used (M4)** | sum (predict_eps) |
| `[DD-2]` denoising scheme | **used (M5)** | SDEdit (forward→t*, subject-conditioned DDIM reverse→0); default t*=200; DDIM 50 steps |
| `[DD-3]` `x₀` / clean target | **used** | preprocessed EEG as x₀ (Phase 1): denoising = SDEdit projection onto the learned EEG prior, NOT verified EOG/EMG removal (honest caveat) |
| `[DD-4]` test-time embedding for unseen subject | **used (M5)** | source embedding e(i) at denoise time |
| `[DD-5]` downstream classifier | **used (M6)** | EEGNet-8,2 (F1=8,D=2,F2=16,kern=64), same config for ICA & SADDPM |
| `[DD-6]` U-Net length | **used** | pad 500→**512**, symmetric zero-pad (6,6), recorded for un-pad |
| `[DD-7]` attention placement | **used** | bottleneck (len 64), 4 heads |
| `[DD-8]` `Z^c,Z^s` features | **used (M4)** | GAP of each decoder's 64-ch penultimate feature -> Linear(64->128), column-mean-centered for L_o |
| `[DD-9]` ArcFace m, κ | **used (M4)** | m=0.5, κ=30; stable cos(θ+m)=cosθ·cosm−sinθ·sinm form (no acos) |
| `[DD-10]` loss weights | **used (M4)** | λ_r=1.0, λ_o=0.1, λ_a=0.1 |
| `[DD-11]` subject-correlation metric | **adapted** | Pearson on a **per-channel log-power-spectrum** descriptor (not trial-mean). **Why:** per-window z-scoring makes the trial mean ≈ 0 and uninformative; the spectral *shape* survives z-scoring and carries subject identity. `trial_mean_descriptor` kept for the literal [DD-11] option. |
| β range | assumed | **used** 1e-4→0.02 (linear) |
| U-Net widths/blocks/norm/dropout | **used** | 64×(1,2,4)=[64,128,256], 2 ResBlocks/level, GN(8), dropout 0.1 |

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
- `tests/`: diffusion unit tests pass (schedule endpoints/monotonicity, q_sample zero-noise + shape,
  predict_xstart inverts q_sample, posterior variance bounds, marginal-match).

## M2 — 1D U-Net (single decoder) + plain DDPM ✓

- `saddpm/models/unet1d.py`: explicit 3-level U-Net (`configs/model.yaml`), **4.44M params**.
  stem 22→64; enc 512→256→128→64 with widths [64,128,256], 2 ResBlocks/level; bottleneck self-
  attention at length 64 (4 heads); symmetric decoder with skip-concat; head 64→22. ResBlock order
  per §4 (GN→SiLU→Conv→+time-emb→GN→SiLU→Dropout→Conv+residual); sinusoidal time emb + MLP(128→512→512).
- `saddpm/diffusion`: added DDPM reverse process (`predict_xstart_from_eps`, `q_posterior_mean`,
  `p_sample`, `p_sample_loop`) taking a generic `eps_fn(x,t)` so it is reused unchanged at M5.
- `saddpm/utils/logging.py`: CSV (always) + optional W&B `RunLogger`. `saddpm/losses/recon.py`: `L_simple`.
- **Overfit-one-batch gate PASSES (Slurm V100, job 840581):** fixed batch (64, 22, 512) from A01-T,
  5000 steps, Adam lr 2e-4. `L_simple` **1.095 → trailing-mean 0.032 (< 0.05 threshold)**. Full
  ancestral sampling (`p_sample_loop`, T=1000) yields EEG-like windows (mean −0.43, std 1.31).
  Figures: `m2_overfit_loss.png`, `m2_overfit_samples.png`.
- `tests/`: **20 total unit tests pass** (+5 model: U-Net full-config shape, timestep-embedding,
  attention shape, p_sample_loop shape, fixed-example memorization).
- W&B note: M2 sanity ran CSV-only (`--wandb` off). Default W&B project `saddpm`; confirm
  entity/project before the first full (M4) training run.

## M6 — EEGNet downstream + ICA baseline (one pair) ✓

- `saddpm/models/eegnet.py` (EEGNet-8,2, [DD-5]); `saddpm/baselines/ica.py` (Infomax ICA on 22 EEG,
  EOG components auto-identified vs the 3 EOG channels via `find_bads_eog`, zeroed, reconstructed,
  then windowed identically — cached on disk); `saddpm/eval/downstream.py` (train EEGNet on denoised
  source-T, test on denoised target-E). ICA & SADDPM share the same EEGNet config (fair).
- **Gate PASSES (V100 job 840615):** A01→A02 end-to-end for both denoisers —
  SADDPM acc **0.272**, ICA acc **0.262** (2 EOG comps excluded/session; 4-class chance 0.25).
  A single off-diagonal cross-subject pair is near chance, as expected; the 9×9 (M7) shows structure.

## M5 — SDEdit denoising + t* sweep ✓

- `GaussianDiffusion.sdedit` ([DD-2]): forward-diffuse a preprocessed segment to t*, then run the
  subject-conditioned DDIM reverse (`predict_eps = ε_θ+ε_φ`) back to 0. Uses the M4 checkpoint.
- **Gate PASSES (V100 job 840613):** denoising 8 held-out A01 Session-E segments, corr(denoised,input)
  **decreases monotonically with t\*** — t*=50:0.996, 100:0.991, 200:0.977, 400:0.919, 600:0.759 —
  i.e. larger t* regularises more toward the learned EEG prior, as SDEdit should. Default t*=200.
- figures: `m5_sdedit_sweep.png` (input vs denoised per t*), `m5_sdedit_corr.png`.

## M3 — subject embeddings + FiLM conditioning ✓ (gate met; weak-conditioning finding logged)

- `saddpm/models/subject_embed.py` (`nn.Embedding(9+1, 128)`, +1 null slot for `--no_subject`/[DD-4]),
  `film.py` (`h'=γ(e)⊙h+δ(e)`, identity-initialised), and `unet1d.py` refactored so FiLM modulates
  every ResBlock when a subject embedding is supplied. Subject embeddings carry weight decay 1e-4
  in a dedicated optimizer param group; cosine LR; AMP; grad clip 1.0.
- **On-disk preprocessing cache** (`saddpm/data/cache.py`, hash `e449a62c`) — all 9 subjects cached
  once (`scripts/preprocess_cache.py`); reused by every GPU job. `WindowDataset`/`pool_subjects`
  added. **DDIM sampler** + **SDEdit** added to the diffusion module.
- **Trained jointly on all 9 subjects' Session-T** (12,690 windows, 80 epochs, Slurm V100 job 840588);
  4.80M params; `L_simple` → ~0.04.
- **Gate (handoff M3: "conditioning changes generated samples per subject"): MET.** From identical
  noise, different subject embeddings give different `x_0` (mean cross-subject correlation **0.985 < 1**).
- **Honest finding (logged):** the conditioning effect is **weak**, and generated samples do **not**
  preserve subject identity — gen-vs-real spectral-descriptor **diagonal dominance = 0.111 = chance**
  (`m3_corr_gen_real.png`). Expected: plain `L_simple` doesn't *require* subject info, so FiLM stays
  near-identity. **This motivates M4** (ArcFace forces `z_s` to encode subject; orthogonality
  disentangles), and identity preservation is properly evaluated at M7 (§8.2).
- `tests/`: subject/FiLM/DDIM/correlation, dual-decoder/ArcFace, EEGNet, SDEdit — all green.
