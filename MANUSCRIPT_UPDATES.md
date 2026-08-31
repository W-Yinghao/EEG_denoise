# Manuscript Update Summary (Phase 2 + honest reconciliation)

Prepared 2026-06-04. This is the delta between the **current `taas_submission/` manuscript** (Phase-1
SADDPM-vs-ICA) and what this codebase has actually established, so the final paper can be assembled from
verified numbers. Honest framing is mandatory (see `RESULTS.md`). **Scope agreed with the authors:** the
paper's thesis is *viability* — "diffusion **can** be used for EEG denoising" + subject-aware — **not**
"diffusion is best." Same-architecture *regression* controls are intentionally not reported.

---

## 0. Three decisions the authors must make first

1. **Which method is "the method"?** The current Method section describes the **Phase-1 generative SADDPM**
   (dual decoder $\epsilon_\theta+\epsilon_\phi$, losses $\mathcal{L}_r/\mathcal{L}_o/\mathcal{L}_a$,
   ArcFace, trained on *clean* signals, denoising = SDEdit). The strong Phase-2 denoising results come from
   a **different model**: a *conditional* denoiser whose U-Net sees `[x_t ; corrupted]` and is trained on
   *paired (corrupted, clean)* data. The final paper should pick the conditional denoiser as the headline
   method (it is what actually denoises well) — or present both clearly. **This is the biggest structural
   change.**
2. **Which evaluation is the headline?** Current: downstream MI classification vs ICA. Recommended: **paired
   ground-truth denoising (EEGdenoiseNet protocol, RRMSE/CC)** — it directly measures denoising and is where
   the positive results live. Downstream/ICA can stay as a secondary Phase-1 study (with corrected numbers).
3. **Phase-1 downstream numbers must be reconciled (see §3.2).** The Tables currently in the manuscript could
   not be reproduced by this codebase; the honest numbers are ~2× lower and reverse the SADDPM>ICA ordering.

---

## 1. NEW content to ADD (Phase 2 — the positive core)

All numbers are RRMSE_temporal↓ / RRMSE_spectral↓ / **CC↑** vs the clean ground truth, EEGdenoiseNet
paired protocol (Zhang et al. 2021). Source: `RESULTS.md` "Phase 2", `results/m8…m12/`.

### 1.1 New method component — the sampling recipe (goes in Method + Experiments)
The conditional denoiser must denoise by **full conditional generation** (start the reverse from pure
noise, condition on the corrupted signal at every step — the Palette/SR3 image-to-image recipe), **not**
by an SDEdit warm-start. With **x0-prediction** (predict the clean signal, not the noise):
- Full generation vs SDEdit warm-start, multi-channel EOG: **CC 0.988 vs 0.787** (warm-start re-injects the
  artifact into the reverse trajectory). Single-channel EOG: 0.901 vs 0.760.
- x0 vs ε parameterization, single-channel EOG: **0.901 vs 0.848**.
- (Negative control) the Phase-1 SDEdit denoiser is **not** a paired-GT denoiser: EOG CC 0.449 < noisy 0.500.

### 1.2 Single-channel EEGdenoiseNet benchmark — *viability* (new table)
| method | EOG CC | EMG CC |
|---|---|---|
| Noisy (input) | 0.500 | 0.506 |
| **SADDPM conditional denoiser (ours)** | **0.901** | **0.680** |
| SimpleCNN (EEGdenoiseNet) | 0.912 | 0.742 |
| ComplexCNN / NovelCNN | 0.909 / 0.878 | 0.691 / 0.831 |

Claim: **competitive with the standard EEGdenoiseNet CNNs on EOG; trails on the harder broadband EMG.**
Do **not** claim superiority. (`results/m10/`)

### 1.3 Multi-channel joint denoising — high fidelity (new table)
| method | EOG CC | EMG CC |
|---|---|---|
| Noisy | 0.662 | 0.542 |
| per-channel EEGdenoiseNet SimpleCNN | 0.844 | 0.853 |
| **SADDPM conditional denoiser (multi-channel, ours)** | **0.988** | **0.994** |

Claim: denoising all 22 channels jointly exploits the artifact's spatial topography, reaching **CC 0.99**,
well above per-channel baselines (which are single-channel by construction). (`results/m9,m11/`)

### 1.4 Subject-aware conditioning is load-bearing — **the main result** (new table + figure)
Subject ablation (denoise with correct vs wrong vs null subject embedding), CC vs clean:
| regime | correct e(s) | wrong e(s′) | null e | **ΔCC (corr−wrong)** |
|---|---|---|---|---|
| shared artifact (baseline) | 0.988 | 0.987 | 0.982 | **+0.001** (inert) |
| **subject-specific artifact** | **0.993** | 0.830 | 0.865 | **+0.163** |
| subject-specific + low SNR | 0.972 | 0.699 | 0.791 | **+0.274** |

- **Robustness:** ΔCC = +0.080 / +0.163 / +0.139 at artifact-specificity gain 0.3 / 0.6 / 1.0; EMG +0.087.
- **Mechanism (state explicitly):** with a *shared* artifact the clean signal is fully recoverable from the
  input, so subject identity is irrelevant (ΔCC≈0). When each subject's artifact occupies a *distinct
  spatial subspace* (physiologically real: anatomy/electrode differences), the denoiser must know *who* the
  subject is to remove the *right* subspace — a wrong embedding actively removes the wrong one (CC 0.993→0.830).
- **Honest condition on the claim:** subject-conditioning is load-bearing **when artifacts are
  subject-specific** (which they physiologically are). State this condition; do not imply it always helps.
  (`results/m12/`)

---

## 2. CHANGES to EXISTING manuscript sections

### 2.1 Method (`sections/method.tex`)
- Add the **conditional denoiser** (input `[x_t ; corrupted]`, x0-prediction, trained on paired data) as the
  denoising model, and describe **full conditional generation** as the inference procedure. The current text
  describes a subject-aware *reverse sampling* of a generative prior — update it to the conditional
  formulation, or clearly separate "generative SADDPM (Phase 1)" from "conditional denoiser (Phase 2)".
- Training config in `sections/experiments.tex` (Adam 1e-4, cosine, batch 64, T=1000 linear, 128-d subject
  embedding, $L_2$ 1e-4) is consistent with the code and can stay.

### 2.2 Experiments — downstream Tables 1–2 ⚠️ **reconcile before submission**
The manuscript reports SADDPM mean **52.8%** vs ICA **50.6%** with within-subject diagonals of **78–92%**.
This codebase's faithful re-implementation (`RESULTS.md` M7, `results/m7/`) gives:
- SADDPM grand mean **0.276**, within-subject diagonal **0.344**; ICA grand mean **0.284**, diagonal **0.393**.
- i.e. **~2× lower** absolute accuracy, near-chance cross-subject, and **SADDPM ≈ ICA (slightly below)** —
  the opposite of the manuscript's SADDPM>ICA ordering.
- **What DID reproduce:** the *lower spread* claim — per-target std **3.3 (SADDPM) vs 4.8 (ICA)** matches.
- **Action:** either replace Tables 1–2 with the reproduced numbers (recommended for honesty) and reframe the
  downstream claim as "SADDPM matches ICA with lower variance across subjects," or document the provenance of
  the current Table numbers if they come from a different (e.g. original-paper) protocol. As written, the
  numbers are not reproducible from this codebase.

### 2.3 Experiments — subject-consistency (Table corr / panel b) ⚠️ reconcile
The manuscript claims SADDPM-*generated* samples preserve subject identity (within > cross correlation).
This codebase (M7) found **generated-vs-real diagonal dominance = 0.11 ≈ chance** — generated samples do
**not** carry subject identity (though subject classification of *real* signals is strong, ArcFace 0.937).
- **Recommended replacement:** drop the "generated samples preserve subject identity" claim and instead use
  the **M12 load-bearing** result, which is the honest and stronger evidence that the subject mechanism is
  real (the embedding demonstrably changes the denoised output when artifacts are subject-specific).

### 2.4 Abstract & Conclusion
- Keep "modest gains over ICA" honesty. **Add** the paired-GT denoising contribution: competitive with
  EEGdenoiseNet CNNs (single-channel), high fidelity multi-channel (CC 0.99), and a subject-conditioning
  mechanism shown to be load-bearing under subject-specific artifacts.
- Conclusion's limitation "compares against a single classical baseline (ICA)" is now **addressed** — we add
  EEGdenoiseNet CNN baselines and a paired ground-truth protocol. Update accordingly.

---

## 3. Deliberately NOT reported (scoping)
- **Same-architecture regression controls** (a regression U-Net matched to the denoiser, and a multi-channel
  conv regressor). They tie the diffusion model on distortion metrics, and a conditioned regressor shows the
  same subject effect. Omitted because the paper claims *viability*, not superiority over regression. Raw data
  retained at `results/m11/`, `results/m12/*_reg_*`, `scripts/m11_mc_baselines.py` for rebuttal only.
- **Honesty guardrail:** because these are omitted, the paper must **not** claim diffusion beats other
  learned denoisers, nor that subject-awareness uniquely requires diffusion. Phrasing must stay at "effective
  / competitive / subject-aware," not "best."

---

## 4. Foreseeable reviewer questions (prepare rebuttal)
1. "Why not a learned regression baseline?" — we have the numbers (parity); decide whether to add on revision.
2. "Subject-specific artifacts are injected by you." — true; it is a physiologically-motivated modeling choice
   (anatomy/electrode differences), and the effect is robust across specificity strength (gain 0.3–1.0).
3. "Multi-channel vs single-channel comparison is uneven." — acknowledged in text; the per-channel CNNs are
   single-channel by construction; the multi-channel result is about exploiting spatial topology.
4. "Downstream gains are modest / near chance cross-subject." — reported honestly; the denoising-fidelity and
   subject-mechanism results are the contribution.

---

## 5. Source-of-truth pointers
- `RESULTS.md` → "Phase 2 (M8–M12)" section: all final tables + honesty ledger.
- `results/m8…m12/`, `results/m9/*_reeval.csv`, `results/unified/`: raw CSVs.
- Code: `saddpm/diffusion/conditional.py`, `saddpm/models/cond_denoiser.py` (full-gen default),
  `scripts/{m10_ablation,m9_reeval,m12_subject_rescue,probe_ensemble}.py`.
