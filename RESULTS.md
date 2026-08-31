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

## M4 — dual decoder + 3 losses + ArcFace ✓

- `saddpm/models/dual_decoder.py`: shared subject-agnostic encoder + content decoder (FiLM e(s),
  `ε_θ`) + individual decoder (no FiLM, `ε_φ`); each decoder GAP→Linear(64→128) → `z_c`/`z_s`
  ([DD-8]); `predict_eps = ε_θ + ε_φ` ([DD-1]). `arcface.py` ([DD-9], m=0.5, κ=30).
- Losses: `L_r`=MSE(ε, ε_θ+ε_φ), `L_o`=‖cross-cov(z_c,z_s)‖²_F, `L_a`=ArcFace CE; weights
  λ=(1.0, 0.1, 0.1) ([DD-10]). **Trained jointly on all 9 (100 epochs, Adam 1e-4 cosine, batch 64,
  AMP, V100 job 840602).**
- **NaN bug found + fixed:** the first run diverged (~epoch 30) because ArcFace's `acos` has an
  infinite gradient at cos=±1 (worse under AMP); replaced with the stable
  `cos(θ+m)=cosθ·cos m − sinθ·sin m` form. Re-run: 0 NaN, smooth convergence.
- **Gate PASSES:** `L_r` 0.86→0.045 (stable), `L_o`≈5e-4; **ArcFace subject-classification accuracy
  on held-out Session-E = 0.937** (chance 0.111) — vs M3's at-chance, confirming the identity losses
  work. Checkpoint `artifacts/checkpoints/m4_saddpm.pt` drives M5–M7. Figure `m4_losses.png`.

## M5 — SDEdit denoising + t* sweep ✓

- `GaussianDiffusion.sdedit` ([DD-2]): forward-diffuse a preprocessed segment to t*, then run the
  subject-conditioned DDIM reverse (`predict_eps = ε_θ+ε_φ`) back to 0. Uses the M4 checkpoint.
- **Gate PASSES (V100 job 840613):** denoising 8 held-out A01 Session-E segments, corr(denoised,input)
  **decreases monotonically with t\*** — t*=50:0.996, 100:0.991, 200:0.977, 400:0.919, 600:0.759 —
  i.e. larger t* regularises more toward the learned EEG prior, as SDEdit should. Default t*=200.
- figures: `m5_sdedit_sweep.png` (input vs denoised per t*), `m5_sdedit_corr.png`.

## M6 — EEGNet downstream + ICA baseline (one pair) ✓

- `saddpm/models/eegnet.py` (EEGNet-8,2, [DD-5]); `saddpm/baselines/ica.py` (Infomax ICA on 22 EEG,
  EOG components auto-identified vs the 3 EOG channels via `find_bads_eog`, zeroed, reconstructed,
  then windowed identically — cached on disk); `saddpm/eval/downstream.py` (train EEGNet on denoised
  source-T, test on denoised target-E). ICA & SADDPM share the same EEGNet config (fair).
- **Gate PASSES (V100 job 840615):** A01→A02 end-to-end for both denoisers —
  SADDPM acc **0.272**, ICA acc **0.262** (2 EOG comps excluded/session; 4-class chance 0.25).

## M7 — full 9×9 sweep + subject correlation ✓ (Phase 1 complete)

Full pairwise protocol (V100 job 840617): for each (source i, target j), denoise i-T and j-E, train
EEGNet on denoised i-T, test on denoised j-E. SADDPM denoise = SDEdit(t*=200, e(i)); ICA denoise =
EOG-component removal. EEGNet config identical for both. Subject correlation per §8.2 on the
spectral descriptor (gen samples from the M4 model, ddim 50 steps).

**Downstream accuracy summary (4-class; chance 0.25):**

| Denoiser | grand mean | diagonal (within-subject) | mean per-target std (spread) |
|----------|-----------|----------------------------|------------------------------|
| **SADDPM** | 0.276 | 0.344 | **0.033** |
| **ICA**    | 0.284 | 0.393 | 0.047 |

SADDPM 9×9 (rows=source, cols=target A01..A09):
```
A01 .44 .27 .35 .29 .23 .29 .26 .26 .27
A02 .28 .30 .25 .25 .23 .25 .24 .28 .27
A03 .30 .25 .47 .24 .25 .28 .28 .31 .33
A04 .26 .24 .27 .27 .27 .25 .27 .22 .28
A05 .29 .28 .29 .25 .27 .25 .25 .24 .25
A06 .25 .26 .27 .26 .27 .29 .28 .25 .25
A07 .39 .26 .28 .28 .24 .27 .32 .27 .27
A08 .27 .26 .33 .25 .24 .25 .29 .38 .32
A09 .29 .26 .25 .25 .24 .24 .27 .25 .34
```
(ICA matrix: `artifacts/runs/m7/acc_ica.csv`; heatmaps `m7_acc_saddpm.png`, `m7_acc_ica.png`.)

**Honest interpretation (this is a clean re-implementation, not a bit-exact reproduction — §13):**
- **Within-subject (diagonal) accuracy is clearly above chance** for both denoisers (SADDPM up to
  0.47, ICA up to 0.60) → the denoising preserves MI-discriminative information.
- **Cross-subject (off-diagonal) ≈ chance (~0.25–0.28)** for both → cross-subject MI transfer is
  genuinely hard on BCI-IV-2a; neither denoiser manufactures transfer that isn't there.
- **ICA is marginally ahead on mean/diagonal; SADDPM has the lower spread** (per-target std
  0.033 vs 0.047) — i.e. SADDPM is more *consistent* across source subjects, matching the paper's
  "lower spread" claim, while not beating ICA on raw accuracy here.
- **Subject correlation (§8.2):** real-vs-real diagonal dominance = **1.00** (panel a, `m7_subjcorr_real.png`);
  SADDPM-generated-vs-real = **0.11 = chance** (panel b, `m7_subjcorr_gen.png`). So the model
  classifies subject identity from *real* signals extremely well (ArcFace 0.937, M4) but its
  *generated* samples do not carry subject-specific spectral signatures — consistent with the weak
  generation-time conditioning observed at M3.
- **Phase-1 denoising caveat ([DD-3], §13):** SADDPM "denoising" is SDEdit projection onto the
  learned EEG prior, **not** verified EOG/EMG removal. The principled paired-ground-truth experiment
  (EEGdenoiseNet, RRMSE/CC) is Phase 2 (M8, not built).

`tests/`: 33 unit tests pass (preprocessing, diffusion + marginal check, U-Net, subject/FiLM/DDIM,
dual-decoder/ArcFace, EEGNet, downstream, subject-correlation).

---

# Phase 2 (M8–M12) — paired-ground-truth denoising on EEGdenoiseNet artifacts

Phase 1 evaluated SADDPM by *downstream classification* (the clean signal is never observed). Phase 2
adds the **paired-ground-truth** protocol of EEGdenoiseNet (Zhang et al. 2021): clean EEG + a known
artifact at controlled SNR, so denoising is scored directly with **RRMSE_temporal / RRMSE_spectral / CC**
against the true clean signal. Two task families:
- **Single-channel** (M8/M10): the standard EEGdenoiseNet benchmark — one channel, EOG or EMG.
- **Multi-channel** (M9/M11/M12): EEGdenoiseNet artifacts injected into the 22-channel BCI-IV-2a windows
  with a physiological spatial topography (ocular = frontal, myogenic = lateral) → paired data *with
  subject labels*, so a subject-conditional denoiser can be trained and the subject embedding tested.

## Headline (Phase 2)

1. **A sampling recipe makes conditional diffusion competitive.** For the noisy-conditioned diffusion
   denoiser, **full conditional generation** (start from pure noise, condition on the corrupted signal at
   every reverse step — Palette/SR3 style) and **x0-parameterization** are decisive. On the standard
   single-channel EOG benchmark this lifts CC **0.838 → 0.901**, reaching parity with the supervised CNNs
   (SimpleCNN 0.912). A conditional-SDEdit warm-start (the Phase-1 instinct) is *worse* — it re-injects
   the artifact into the reverse trajectory (CC 0.760 at t\*=400 vs 0.901 full-gen).
2. **Multi-channel joint denoising reaches high fidelity.** Denoising all 22 channels jointly lets the
   subject-conditional diffusion model exploit the artifact's spatial topography, reaching CC **0.99**;
   the standard EEGdenoiseNet CNNs, which are single-channel by construction, reach ~0.85 on the same data.
3. **Subject-aware conditioning is load-bearing when artifacts are subject-specific.** With a *shared*
   artifact, the subject embedding is inert (correct ≈ wrong e(s), ΔCC ≈ 0). With *subject-specific*
   artifact topographies (physiologically real: anatomy/electrode differences), the embedding becomes
   essential — correct e(s) CC 0.993 vs **wrong** e(s′) 0.830 (ΔCC up to **+0.27**), robust across
   artifact-specificity strength and present for EOG and EMG.

## M8 — single-channel EEGdenoiseNet scaffolding (overall CC over SNR −7…2 dB)

`scripts/m8_benchmark.py` (`results/m8/`). All methods on identical paired data.

| method | EOG CC | EMG CC | note |
|--------|--------|--------|------|
| Noisy (input) | 0.500 | 0.506 | reference |
| SDEdit (unconditional prior) | 0.449 | 0.579 | **fails** on EOG (< noisy) — Phase-1 method is not a denoiser |
| CondDiff (eps, original) | 0.842 | 0.623 | noisy-conditioned DDPM |
| SimpleCNN / ComplexCNN | 0.913 / 0.909 | 0.740 / 0.691 | EEGdenoiseNet CNN baselines |
| NovelCNN | 0.878 | 0.831 | |

**Audit fix (B1):** M8 sampled the diffusion nets without `.eval()`, leaving dropout active across the 50
DDIM steps — degrading only the diffusion methods (the CNNs/M9 use `.eval()`). Fixed in `m8_benchmark.py`
(`_cond_denoise`/`_sdedit_denoise`). This + the recipe below is isolated in M10.

## M9 — multi-channel subject-conditional denoiser + corrected sampler (`scripts/m9_reeval.py`)

`SubjectConditionalDenoiser`: U-Net sees `[x_t ; corrupted]`, FiLM on the subject embedding, x0-prediction,
trained on synthetic `(corrupted, clean)` pairs. **The default conditional-SDEdit start (t\*=400) was badly
suboptimal**; switching to full conditional generation (`t_star=None`, now the default in
`saddpm/models/cond_denoiser.py`) is a large, retraining-free gain (`results/m9/{EOG,EMG}_reeval.csv`):

| sampler | EOG correct e(s) | wrong e(s′) | null e | EMG correct | wrong | null |
|---------|------------------|-------------|--------|-------------|-------|------|
| t\*=400 (old) | 0.787 | 0.786 | 0.777 | 0.994 | 0.993 | 0.978 |
| **full-gen (new)** | **0.988** | 0.987 | 0.982 | **0.994** | 0.993 | 0.977 |

t\* sweep (probe, `scripts/probe_ensemble.py`, EOG): 0.787 (400) → 0.935 (600) → 0.986 (800) → 0.987 (full).
**Posterior-mean ensembling was tested and refuted** — averaging K reverse draws is flat (CC 0.7397 for
K=1→16); the conditional sampler is already near the posterior mean. (Note: at the old t\*=400 the
embedding ablation is ≈0 — the basis for M12.)

## M10 — single-channel ablation: what makes conditional diffusion competitive (`results/m10/`)

One U-Net per parameterization; other arms are inference-time toggles; all sampled in `.eval()`.

| arm | EOG CC | EMG CC | isolates |
|-----|--------|--------|----------|
| A1 eps, train-mode (dropout on) | 0.838 | 0.658 | reproduces original CondDiff |
| A2 eps, eval-mode | 0.848 | 0.659 | **B1 dropout bug** (+0.010) |
| A3 **x0** parameterization | 0.901 | 0.679 | **eps → x0** (+0.053, the main lever) |
| A4 + EMA | 0.901 | 0.680 | EMA (≈0 here) |
| A5 full-gen (best) / t\*=400 | **0.901** / 0.760 | 0.680 / 0.688 | **full-gen ≫ warm-start** (EOG) |
| A6 + unit-variance-clean | 0.893 | 0.675 | normalization (no help) |
| **R1 SimpleCNN / R2 ComplexCNN** | **0.912 / 0.909** | **0.742** / 0.691 | supervised reference |
| Noisy | 0.500 | 0.506 | floor |

**Read:** the recipe (x0 + full-gen) takes EOG to **parity** with the CNNs (0.901 vs 0.912) and is monotone
and mechanistic. EMG is harder (broadband): diffusion 0.680 stays **behind** SimpleCNN 0.742. Competitive,
not winning — honest supporting result.

## M11 — multi-channel denoising fidelity (`results/m11/`)

Multi-channel joint denoising vs per-channel baselines, identical test windows as M9.

| method | EOG CC | EMG CC |
|--------|--------|--------|
| Noisy (input) | 0.662 | 0.542 |
| per-channel EEGdenoiseNet SimpleCNN | 0.844 | 0.853 |
| **SADDPM-Cond (diffusion, full-gen)** | **0.988** | **0.994** |

**Read:** denoising the 22 channels jointly (exploiting the artifact's spatial topography) reaches CC 0.99,
well above the per-channel EEGdenoiseNet CNNs (0.85), which operate one channel at a time by construction.

## M12 — subject-aware: making the embedding load-bearing (`results/m12/`)

`scripts/m12_subject_rescue.py`. Subject ablation (correct vs wrong vs null e(s)) under each regime, EOG
unless noted. Verdict: ΔCC(correct − wrong) ≥ 0.02 ⇒ subject identity is load-bearing.

| regime | correct e(s) | wrong e(s′) | null e | **ΔCC (corr−wrong)** |
|--------|--------------|-------------|--------|----------------------|
| baseline (−7…2, shared artifact) | 0.988 | 0.987 | 0.982 | +0.001 ❌ |
| low-SNR (−16…−6, shared) | 0.966 | 0.964 | 0.962 | +0.002 ❌ |
| **subject-specific artifact** (gain 0.6) | **0.993** | 0.830 | 0.865 | **+0.163 ✅** |
| subject-specific + low-SNR | 0.972 | 0.699 | 0.791 | **+0.274 ✅** |

*"wrong e(s′)" is the mean over **all** wrong subject embeddings (the off-diagonal of the embedding-swap
matrix, Fig. F3) = 0.830; the per-window single-random-wrong estimate in `results/m12/EOG_subjart.csv` is
0.828. The two agree to rounding; we report the swap-matrix value for figure/table consistency.*

**Robustness** (subject-specific artifact): ΔCC = +0.080 / +0.163 / +0.139 at topo-gain 0.3 / 0.6 / 1.0;
EMG +0.087. Load-bearing across artifact-specificity strengths and both artifact types.

**Mechanism (honest):** additive, fully-observed denoising with a *shared* artifact is subject-agnostic —
the clean signal is already in the corrupted window, so the model needs no identity. Subject identity only
matters when it carries information the window does not: when each subject's artifact occupies a distinct
spatial subspace, the denoiser must know *who* the subject is to remove the *right* subspace (a wrong
embedding actively removes the wrong one → CC collapses 0.993 → 0.830; even null > wrong). Low SNR alone
does **not** induce this. This supports a "Subject-Aware" claim honestly — *conditioned on subject-specific
artifacts being present*, which is physiologically the case.

### Phase-2 honesty ledger
- SDEdit (Phase-1 denoiser) is not a paired-GT denoiser (EOG CC 0.449 < noisy); reported as a negative control.
- The contribution is **viability**: the diffusion denoiser is **competitive** with the standard EEGdenoiseNet
  CNNs on single-channel EOG (0.901 vs 0.912) and **trails** on the harder EMG (0.680 vs 0.742); the paper does
  not claim diffusion is the best denoiser, only that it is an effective and subject-aware one.
- The multi-channel and subject-aware gains are demonstrated on **semi-synthetic** data (artifacts injected
  with modeled topographies), as is standard for EEGdenoiseNet-style paired evaluation; the subject-specific
  topography is a physiologically-motivated modeling choice.
- `unified_compare.py`'s earlier SADDPM-Cond CC 0.34 was a stale-code/checkpoint mismatch (old eps code +
  x0 checkpoint), not a real result; the probe/M9-reeval numbers (0.99) are authoritative.
