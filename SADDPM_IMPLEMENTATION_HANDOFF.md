# SADDPM Implementation Handoff — Build-From-Zero Guide

**Audience:** a fresh Claude Code instance on a GPU server with **no prior context** about this project.
**Goal:** implement, train, and evaluate **SADDPM** (Subject-Aware Denoising Diffusion Probabilistic Model) for cross-subject EEG denoising on **BCI Competition IV-2a**, reproducing the paper's experimental *structure* and producing honest, logged results.

> **Read this entire document before writing any code.** It is self-contained. Where the source paper is ambiguous, this guide makes an explicit **DESIGN DECISION** (tagged `[DD-n]`) with a default and rationale. Follow the defaults, **log them**, and do **not** silently invent alternatives. The full list is in §12 (Assumptions Ledger).

---

## 0. TL;DR — what you are building

A subject-conditional diffusion model with a **dual decoder** (content + individual) and three losses, used to denoise motor-imagery EEG, evaluated by **downstream 4-class classification** across subjects and by a **subject-correlation** analysis. Pipeline:

```
raw BCI-IV-2a  →  preprocess  →  train SADDPM (subject-conditional DDPM, dual decoder)
                                      │
                test segment  →  SDEdit denoise (forward to t*, subject-conditioned reverse to 0)
                                      │
                       denoised EEG  →  EEGNet downstream classifier  →  9×9 cross-subject accuracy matrix
                                      └→ subject-correlation matrices (Table 3 analogue)
baseline: ICA denoise → same EEGNet
```

There are **two phases**:
- **Phase 1 (core, faithful to paper):** SADDPM + SDEdit + EEGNet downstream + ICA baseline on BCI-IV-2a. Produces the 9×9 accuracy matrices and the subject-correlation matrices.
- **Phase 2 (recommended extensions, strengthen the paper):** paired synthetic-artifact denoising with ground truth (EEGdenoiseNet) for real denoising metrics (RRMSE/CC), DL denoising baselines (CNN/GAN autoencoders), and ablations of the subject-aware components. Build Phase 1 first.

---

## 1. The method (self-contained background)

### 1.1 DDPM essentials
Forward (variance-preserving) diffusion adds Gaussian noise over `T` steps to a clean signal `x₀ ∈ ℝ^{C×L}`:

- single step: `q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜ I)`
- closed form marginal: `q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ) I)`, with `αₜ = 1-βₜ`, `ᾱₜ = ∏_{s≤t} αₛ`
- reparameterization (use this for training): `xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε`, `ε ~ N(0, I)`

Training objective (noise prediction): `L_simple = E_{t,x₀,ε}[ ‖ε − ε_θ(xₜ, t)‖² ]`.

### 1.2 Subject-aware extension (the contribution)
- Each subject `s ∈ {1..N}` (N=9) has a **learnable embedding** `e(s) ∈ ℝ^d`, d=128.
- Conditioning is injected at multiple U-Net blocks via **FiLM**: `h' = γ(e) ⊙ h + δ(e)`, where `γ, δ` come from a small MLP on `e`.
- **Dual decoder** on a shared encoder (see Fig. in paper):
  - **content (denoised-signal) decoder** `ε_θ(xₜ, t, s)` — subject-conditioned, predicts the clean-signal noise component.
  - **individual-difference decoder** `ε_φ(xₜ, t)` — captures subject-specific variation.
  - reverse-step noise estimate = `ε_θ + ε_φ`  `[DD-1]`.
  - **classifier head** `ω` on the individual branch's pooled feature → subject logits (ArcFace).

### 1.3 Losses
Let `Z^c, Z^s ∈ ℝ^{B×d}` be the **column-mean-centered** pooled features of the content / individual decoders over a batch of size B; `y` = ground-truth subject.

- **Reverse / reconstruction:** `L_r = E[ ‖ε − (ε_θ(xₜ,t,s) + ε_φ(xₜ,t))‖² ]`
- **Orthogonality (disentangle content vs subject):** `L_o = ‖ (Z^c)ᵀ Z^s ‖_F²`  (zero ⇔ empirical cross-covariance is zero)
- **Subject identification (ArcFace):** `L_a = −E[ log( e^{κ·cos(θ_y+m)} / ( e^{κ·cos(θ_y+m)} + Σ_{j≠y} e^{κ·cos θ_j} ) ) ]`
- **Total:** `L = λ_r L_r + λ_o L_o + λ_a L_a`

> **Math sanity (do this):** verify the closed-form marginal numerically — sample `xₜ` two ways (iterated single steps vs. one-shot reparameterization) for several `t` and check the empirical mean/cov match `√ᾱₜ x₀` and `(1-ᾱₜ)I`. This catches schedule bugs immediately.

---

## 2. Critical conceptual decisions (READ — these are the gaps the paper leaves open)

The source paper is **underspecified** on three conceptual points. Defaults below are chosen to be standard, defensible, and runnable. **Use them, log them, and surface them to the user in your first status report.**

### `[DD-2]` How does a *diffusion* model "denoise" a real EEG segment?
A vanilla DDPM reverse process *generates from pure noise*; it does not, by itself, map a given noisy signal to a clean one. **Default: SDEdit-style denoising.**
- Forward-diffuse the (preprocessed) test segment `y` to a moderate step `t*`: `x_{t*} = √ᾱ_{t*} y + √(1-ᾱ_{t*}) ε`.
- Run the subject-conditioned reverse process from `t*` down to `0` to get `x̂₀` (the "denoised" output).
- **`t*` controls strength:** default `t* = 0.2·T = 200`. (Smaller ⇒ closer to input; larger ⇒ more regularized.) Treat `t*` as a hyperparameter; report it.

### `[DD-3]` What is `x₀` (the "clean target") given that BCI-IV-2a has no ground-truth clean EEG?
**Default (Phase 1):** treat the preprocessed EEG segment as `x₀`; the diffusion model learns the *data distribution*, and "denoising" = SDEdit projection onto that learned manifold (removes off-manifold high-frequency/atypical content). **Be explicit in results:** this denoises toward the learned EEG prior, not toward a verified artifact-free signal.
**Phase 2 (recommended, gives real denoising ground truth):** use **EEGdenoiseNet** (clean EEG + EOG/EMG templates) to build paired `(noisy = clean + artifact @ SNR, clean)` data; train/eval with **RRMSE** and **correlation coefficient** against the true clean signal. This is the principled denoising experiment and directly answers reviewers.

### `[DD-4]` Which subject embedding is used at test time on a subject whose embedding was **not** trained?
Subject embeddings exist only for **training** subjects — a genuine tension in the method for unseen targets.
**Default (Phase 1, pairwise protocol "train on source `s_i`, test on target `s_j`"):** at inference use the **source subject's embedding** `e(s_i)` (the model only knows source subjects). Also implement a `--no_subject` mode (content branch with a null/averaged embedding) for an ablation.
**Phase 2:** *embedding adaptation* — freeze the network, optimize a fresh `e` on the target's **unlabeled** data for a few hundred steps via `L_simple`, then denoise. (This is the honest way to get a subject-aware benefit on unseen subjects; worth reporting.)

### `[DD-5]` What downstream classifier produced the accuracies? (paper never says)
**Default: EEGNet** (Lawhern et al., 2018; `EEGNet-8,2`), the standard BCI-IV-2a baseline; public, reproducible. Train **per source subject** on its denoised Session-T, evaluate on the (denoised) target Session-E. Use the **same** classifier + training config for ICA and SADDPM to keep the comparison fair. (Classic alternative: FBCSP + LDA — implement only if asked.)

---

## 3. Data: BCI Competition IV-2a

- **Source:** load via **MOABB** dataset `BNCI2014_001` (this *is* BCI-IV-2a), which downloads + parses the GDF files; use **MNE** for preprocessing. (Fallback: raw GDF from the BNCI Horizon 2020 archive.) Requires internet on first run — if the server is offline, pre-download the cache and copy it.
- **Structure:** 9 subjects (A01–A09), 2 sessions each (`T` train / `E` test), 6 runs × 48 trials = 288 trials/session, 4 classes (left hand, right hand, both feet, tongue), 22 EEG + 3 EOG channels, 250 Hz.
- **Preprocessing (paper-given — implement exactly):**
  1. keep the **22 EEG** channels; drop reference and the **3 EOG** channels (keep EOG separately for the ICA baseline).
  2. **band-pass** FIR **1–50 Hz**.
  3. **notch** 50 Hz.
  4. resample to 250 Hz (already 250; keep the step for generality).
  5. **sliding window**: length **2 s (500 samples)**, step **0.5 s**.
  6. **per-channel z-score** (zero mean, unit variance) within each window.
- `[DD-6]` **Length for the U-Net:** pad/crop each 500-sample window to **512** so 3× downsampling is clean (512→256→128→64). Record the pad and remove it on output.
- **Labels:** keep the 4-class MI label per window (for downstream eval) and the subject id (for SADDPM conditioning + ArcFace).

---

## 4. Model architecture (concrete spec)

Input tensor: `x ∈ ℝ^{B×22×512}`.

**Time embedding:** sinusoidal(`t`) (dim 128) → MLP(128→512→512) with SiLU. Added inside every ResBlock.
**Subject embedding:** `nn.Embedding(N=9, 128)`; per-FiLM-site MLP(128 → 2·channels) producing `(γ, δ)`.

**Shared encoder (1D U-Net down path):**
- stem: `Conv1d(22 → 64, k=5, p=2)`
- 3 levels, base channels 64, **channel mults (1, 2, 4)** ⇒ widths [64, 128, 256]; **2 ResBlocks/level**; downsample with stride-2 `Conv1d` (length 512→256→128→64).
- **ResBlock:** `GroupNorm(8) → SiLU → Conv1d(k3,p1) →` add time emb `→` **FiLM(subject)** `→ GroupNorm(8) → SiLU → Dropout(0.1) → Conv1d(k3,p1)` + residual (1×1 conv if channels change).
- **Self-attention** (multi-head, heads=4) at the **bottleneck** (length 64) `[DD-7]` (optionally also at length 128).

**Two decoders (mirror of encoder, with skip connections):**
- **content decoder** `D_c`: subject-conditioned (FiLM uses `e(s)`), up path 256→128→64, head `GroupNorm→SiLU→Conv1d(64→22)` ⇒ `ε_θ ∈ ℝ^{B×22×512}`.
- **individual decoder** `D_s`: **not** subject-conditioned (FiLM disabled / identity), same shape ⇒ `ε_φ`.
- pooled features for losses: global-average-pool the penultimate (64-ch) feature of each decoder ⇒ `z^c, z^s ∈ ℝ^{128}` (project 64→128 with a linear if needed) `[DD-8]`.
- **classifier** `ω`: ArcFace head on `z^s` over N=9 subjects (margin `m=0.5`, scale `κ=30`) `[DD-9]`.

> Keep all sizes in a `config` (dataclass / YAML). Do not hard-code.

---

## 5. Diffusion + losses (spec)

- **Schedule:** `T=1000`, **linear** β from `1e-4` to `0.02`. Precompute `α, ᾱ, √ᾱ, √(1-ᾱ)` as buffers.
- **Training step:**
  1. sample batch `(x₀, s, y_mi)`; sample `t ~ U{1..T}`, `ε ~ N(0,I)`.
  2. `xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε`.
  3. forward: `ε_θ = D_c(E(xₜ), t, s)`, `ε_φ = D_s(E(xₜ), t)`, `z^c, z^s`, `logits = ω(z^s)`.
  4. `L_r = mse(ε, ε_θ + ε_φ)`; `L_o = ‖(centered z^c)ᵀ(centered z^s)‖_F²` over the batch; `L_a = arcface(logits, s, m, κ)`.
  5. `L = λ_r L_r + λ_o L_o + λ_a L_a`; backprop; update network + embeddings.
- **Weights `[DD-10]`:** `λ_r=1.0, λ_o=0.1, λ_a=0.1` (starting point; expose as flags).
- **Embedding regularization (paper-given):** weight decay `1e-4` on the subject embeddings only.

---

## 6. Training (spec)

- Optimizer **Adam**, lr `1e-4`, **cosine annealing** to 0 over training.
- Batch **64**, **100 epochs** over the pooled training windows.
- Mixed precision (`torch.cuda.amp`) ok; gradient clip 1.0.
- **Reproducibility:** seed everything (torch, numpy, python, cudnn deterministic where feasible); log the seed.
- **Logging:** Weights & Biases (the user uses W&B) — log `L, L_r, L_o, L_a`, lr, and periodic generated/denoised samples. Also write a local CSV.
- **Checkpoints:** save best + last; store config + git commit hash in the checkpoint.
- **What set to train on:** for the pairwise protocol, train one SADDPM **per source subject** on that subject's Session-T windows (its embedding is trainable). (Also support a "train on all subjects jointly" mode for Phase-2 ablations.)

---

## 7. Inference / denoising (SDEdit) — spec

```
def sdedit_denoise(model, y, subject_id_for_embedding, t_star=200):
    # y: preprocessed segment [22, 512], already z-scored
    eps = randn_like(y)
    x = sqrt_abar[t_star]*y + sqrt_one_minus_abar[t_star]*eps      # forward to t*
    for t in reversed(range(0, t_star)):                            # subject-conditioned reverse
        eps_pred = model.eps_theta(x, t, subject_id_for_embedding) + model.eps_phi(x, t)
        x = ddpm_posterior_step(x, eps_pred, t)                     # standard DDPM x_{t}→x_{t-1}
    return x                                                        # denoised x̂₀
```
- Implement the standard DDPM posterior mean/variance step (Ho et al.). DDIM sampling optional for speed.
- For the pairwise protocol, `subject_id_for_embedding = source subject` `[DD-4]`.

---

## 8. Evaluation protocol

### 8.1 Downstream classification (Tables 1 & 2 analogue)
- For each **(source `i`, target `j`)** pair: take SADDPM trained on `i`; denoise `i`'s Session-T and `j`'s Session-E with SDEdit (using `e(i)`); train EEGNet on denoised `i`-Session-T (4-class), test on denoised `j`-Session-E. Record accuracy ⇒ **9×9 matrix**; the diagonal is within-subject; add a **mean row** (column average over sources).
- **ICA baseline:** identical pipeline but denoise with ICA instead of SADDPM. ICA via MNE (Infomax/Picard); identify EOG components by correlation with the 3 EOG channels; zero them; reconstruct. Same EEGNet config.
- Report both matrices + the grand mean and the per-target std (the paper claims SADDPM has lower spread).

### 8.2 Subject-correlation (Table 3 analogue) `[DD-11]`
- Define a per-subject signal descriptor: the **trial-averaged** segment per channel (or band-power feature vector). Default: trial-mean signal, flattened over channels×time.
- `corr(i, j)` = **Pearson** correlation between subject `i`'s and subject `j`'s descriptors (average over channels if computed per-channel).
- Panel (a): real-vs-real (within BCI-IV-2a). Panel (b): SADDPM-**generated** samples of subject `i` vs real subject `j`. Diagonal should dominate ⇒ subject identity preserved.

### 8.3 Phase-2 denoising metrics (if EEGdenoiseNet added)
- RRMSE (time + spectral) and correlation coefficient against ground-truth clean signal, across input SNRs; compare SADDPM vs ICA vs CNN/GAN autoencoder.

---

## 9. Repository structure

```
saddpm/
  configs/            # YAML configs (data, model, train, eval)
  saddpm/
    data/             # moabb/mne loading, preprocessing, windowing, datasets
    models/           # unet1d.py, film.py, subject_embed.py, dual_decoder.py, arcface.py, eegnet.py
    diffusion/        # schedule.py, gaussian_diffusion.py (q_sample, p_sample, sdedit)
    losses/           # recon.py, orthogonality.py, arcface_loss.py
    baselines/        # ica.py
    eval/             # downstream.py (9x9), subject_corr.py
    utils/            # seed, logging (wandb+csv), checkpoint
  scripts/
    train_saddpm.py
    denoise.py
    eval_downstream.py
    eval_subject_corr.py
    run_pairwise_matrix.py   # orchestrates the 9x9 sweep
  tests/              # unit tests + math sanity checks
  environment.yml / requirements.txt
  README.md           # how to run, reproduces results
  RESULTS.md          # logged numbers + assumptions actually used
```

---

## 10. Environment

- Python ≥3.10, **PyTorch ≥2.1 (CUDA)**, `mne`, `moabb`, `numpy`, `scipy`, `scikit-learn`, `einops`, `pyyaml`, `tqdm`, `wandb`, `matplotlib`.
- Create a conda env or venv; pin versions in `environment.yml`/`requirements.txt`.
- Detect GPU; fail loudly if CUDA is unavailable (training is heavy: T=1000, 9 models).
- First step: a `scripts/check_env.py` that prints device, versions, and confirms the dataset downloads/parses.

---

## 11. Build milestones (do them in order; verify each before moving on)

1. **M0 — env + data:** `check_env.py` passes; load one subject via MOABB; print shapes; plot one preprocessed window. Commit.
2. **M1 — diffusion core:** schedule + `q_sample`; **numerical marginal sanity check** passes (§1.3). Unit-tested. Commit.
3. **M2 — U-Net (single decoder, no subject):** train plain DDPM (`L_simple`) on one subject; **overfit a single batch** (loss → ~0); generate samples that look EEG-like. Commit.
4. **M3 — subject conditioning:** add embeddings + FiLM; confirm conditioning changes generated samples per subject. Commit.
5. **M4 — dual decoder + 3 losses + ArcFace:** full SADDPM trains stably; subject-classification accuracy on the ArcFace head > chance. Commit.
6. **M5 — SDEdit denoise:** denoise held-out segments; sweep `t*`; visualize input vs denoised. Commit.
7. **M6 — EEGNet downstream + ICA baseline:** single (source,target) pair end-to-end for both denoisers. Commit.
8. **M7 — full 9×9 sweep + subject-correlation:** produce both accuracy matrices, mean row, grand mean, std; produce the two correlation matrices. Write `RESULTS.md`. Commit.
9. **M8 — (optional) Phase 2:** EEGdenoiseNet paired denoising + DL baselines + ablations (no-subject, no-`L_o`, no-`L_a`, embedding-adaptation).

At each milestone: run tests, commit with a clear message, and post a short status (what passed, any deviation from this doc).

---

## 12. Assumptions ledger (paper-given vs assumed)

| Item | Source | Value |
|------|--------|-------|
| Dataset, channels, sessions, classes, sample rate | **paper** | BCI-IV-2a, 22 EEG (+3 EOG), T/E, 4-class, 250 Hz |
| Band-pass / notch / window / z-score | **paper** | 1–50 Hz FIR, 50 Hz notch, 2 s/0.5 s, per-channel z-score |
| Diffusion T, β type, objective | **paper** | T=1000, linear β, ε-prediction (`L_simple`) |
| Subject embedding dim, emb. weight decay | **paper** | 128, 1e-4 |
| Backbone family, FiLM, time emb, attention, dual decoder, 3 losses | **paper** | 1D-Conv U-Net; FiLM; sinusoidal; self-attn; content+individual+classifier; `L_r,L_o,L_a` |
| Optimizer/lr/schedule/batch/epochs | **paper** | Adam, 1e-4, cosine, 64, 100 |
| `[DD-1]` reverse noise = `ε_θ+ε_φ` | assumed | sum |
| `[DD-2]` denoising scheme | assumed | SDEdit, `t*`=200 |
| `[DD-3]` `x₀` / clean target | assumed | preprocessed EEG as `x₀` (Phase 1); EEGdenoiseNet pairs (Phase 2) |
| `[DD-4]` test-time embedding for unseen subject | assumed | source embedding (Phase 1); embedding adaptation (Phase 2) |
| `[DD-5]` downstream classifier | assumed | EEGNet-8,2 |
| `[DD-6]` U-Net length | assumed | pad 500→512 |
| `[DD-7]` attention placement | assumed | bottleneck (len 64) |
| `[DD-8]` `Z^c,Z^s` features | assumed | GAP of decoder penultimate (dim 128), column-mean-centered |
| `[DD-9]` ArcFace m, κ | assumed | 0.5, 30 |
| `[DD-10]` loss weights | assumed | λ_r=1, λ_o=0.1, λ_a=0.1 |
| `[DD-11]` subject-correlation metric | assumed | Pearson on trial-mean descriptor |
| β range | assumed | 1e-4→0.02 |
| U-Net widths/blocks/norm/dropout | assumed | 64×(1,2,4), 2 blocks, GN(8), 0.1 |

Keep this table in `RESULTS.md` and **update it with the values you actually used**.

---

## 13. Known limitations / honest notes (carry into the paper)

- Results here will not exactly equal the manuscript's numbers — several settings above are reconstructions of unspecified choices. Treat this as a clean re-implementation, not a bit-exact reproduction.
- The Phase-1 "denoising" is projection onto a learned EEG prior, **not** verified removal of EOG/EMG; the honest denoising claim needs Phase 2 (paired ground truth).
- Subject embeddings do not transfer to unseen subjects without `[DD-4]` adaptation — state this clearly.
- If the user has the **original implementation**, prefer aligning to it over these defaults; flag any conflict.

---

## 14. Server-side kickoff prompt (paste this into Claude Code on the server)

> Place this file at the repo root as `SADDPM_IMPLEMENTATION_HANDOFF.md`, then start Claude Code and paste the prompt below.

```
You are setting up a research codebase from scratch on this GPU server. Read the file
SADDPM_IMPLEMENTATION_HANDOFF.md in full BEFORE doing anything else; it is the
authoritative spec and is self-contained.

Project: implement, train, and evaluate SADDPM (a subject-aware diffusion model) for
cross-subject EEG denoising on BCI Competition IV-2a, per that document.

How I want you to work:
1. First, write a short plan (phases + milestones M0–M7 from §11) and an `environment.yml`,
   and create the repo structure in §9. Initialize git; commit.
2. Build STRICTLY milestone by milestone (M0→M7). After each milestone: run its sanity
   check / tests, commit with a clear message, and give me a 3–5 line status (what passed,
   any deviation from the doc). Do NOT skip ahead.
3. Follow every DESIGN DECISION default in §2/§12 exactly. Do NOT silently invent
   alternatives. If a default is infeasible on this machine (no internet, no GPU, missing
   data), STOP and ask me rather than guessing.
4. Keep all hyperparameters in YAML/dataclass configs (no magic numbers). Seed everything
   for reproducibility. Log to Weights & Biases AND a local CSV. Maintain RESULTS.md with the
   assumptions ledger (§12) updated to the values you actually used.
5. Verify the diffusion math numerically (the forward-marginal check in §1.3) at M1, and
   overfit a single batch at M2, before scaling up.
6. Engineering standards: type hints, docstrings, unit tests in tests/, small focused
   commits, no dead code.

Environment notes for this server: <FILL IN: conda/module setup, GPU type, data path or
whether internet is available for MOABB download, W&B entity/project, where to store
checkpoints>.

Start with M0 (env + data): create environment.yml, scripts/check_env.py, load one subject
via MOABB, print shapes, plot one preprocessed window. Then stop and show me the result
before continuing.
```

**Before pasting, fill the `<...>` server notes** (conda env name, GPU, whether the server has internet for the MOABB download or if you must pre-stage the dataset, your W&B entity/project, checkpoint directory). Everything else is specified in this document.
```

---

### How to use this handoff
1. Copy `SADDPM_IMPLEMENTATION_HANDOFF.md` to your server repo root.
2. (Optional but useful) also copy the LaTeX paper folder `taas_submission/` so the server agent can cross-read the method/figure.
3. Pre-stage the BCI-IV-2a data if the server is offline (run MOABB `BNCI2014_001` once on a machine with internet and copy the `~/mne_data` / MOABB cache).
4. Start Claude Code on the server, fill the `<...>` notes in §14's prompt, and paste it.
