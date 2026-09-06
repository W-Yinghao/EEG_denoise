# Figure Guide

All figures live in `artifacts/figures/paper/` as both `.pdf` (use in LaTeX) and `.png` (preview).
Regenerate any with `python scripts/figgen_plot.py --which all` after the eval data exists. Numbers below
are the actual values in the rendered figures (EOG unless noted). Target venue is TAAS `acmsmall`
(single-column); every figure is sized to the text width.

## At-a-glance placement

| Fig | File (`.pdf`) | Paper section | Role | Main / Appendix |
|-----|---------------|---------------|------|-----------------|
| **F2** | `F2_metrics_vs_snr_EOG` | Experiments → quantitative benchmark | the EEGdenoiseNet-style headline plot (viability) | **Main** |
| **F1** | `F1_waveforms_EOG` | Experiments → qualitative results | intuitive "it denoises" example traces | **Main** |
| **F4** | `F4_sampling_recipe` | Experiments → method analysis / ablation | why the method works (sampling recipe) | **Main** |
| **F5** | `F5_multichannel_EOG` | Experiments → multi-channel | spatial topography + joint denoising | **Main** |
| **F3** | `F3_subject_swap_heatmap` | Experiments → subject-aware (the climax) | **the subject-aware contribution** | **Main (centerpiece)** |
| **F6** | `F6_psd_EOG` | Experiments → qualitative (beside F1) or Appendix | spectral recovery | Main-or-Appendix |
| F1/F2/F5/F6 **EMG** variants | `*_EMG` | same subsections | EMG counterpart | **Appendix** (or stacked under EOG) |

Recommended **main-text figure order** (narrative): F2 → F1 (+F6) → F4 → F5 → **F3**. This goes
quantitative-benchmark → qualitative-proof → why-it-works → spatial-scaling → the subject-aware result as
the climax. Keep `sections/fig_architecture.tex` (the model schematic) in the Method section as before.

---

## Per-figure detail

### F2 — Denoising metrics vs input SNR  (`F2_metrics_vs_snr_{EOG,EMG}`)
- **Shows:** RRMSE_temporal, RRMSE_spectral, CC vs input SNR (−7…+2 dB) for the SADDPM conditional denoiser
  and the EEGdenoiseNet CNNs (SimpleCNN/ComplexCNN/NovelCNN), with the no-op "Noisy" reference (gray).
- **Takeaway:** our denoiser **tracks the supervised CNNs** across all SNRs (CC ~0.80 at −7 dB up to ~0.94 at
  +2 dB), establishing diffusion as a *viable* EEG denoiser. This is EEGdenoiseNet's signature comparison plot.
- **Placement:** Experiments, first results figure (single-channel benchmark). Full text width.
- **Draft caption:** *"Single-channel denoising on the EEGdenoiseNet benchmark. RRMSE\textsubscript{temporal},
  RRMSE\textsubscript{spectral} and correlation coefficient (CC) versus input SNR for SADDPM and the
  EEGdenoiseNet CNN baselines; ``Noisy'' is the unprocessed input. SADDPM is competitive with the supervised
  CNNs across the SNR range."*

### F1 — Example denoising waveforms  (`F1_waveforms_{EOG,EMG}`)
- **Shows:** time-domain traces at four SNR levels — contaminated (red), clean ground truth (black), SADDPM
  denoised (blue).
- **Takeaway:** the denoised trace closely follows the clean signal while the large ocular drift (most visible
  ~1.5–2 s) is removed; the standard intuitive EEGdenoiseNet result figure.
- **Placement:** Experiments, qualitative results (right after / beside F2). Text width.
- **Draft caption:** *"Example EOG-contaminated segments at four input SNRs: the SADDPM-denoised signal (blue)
  recovers the clean ground truth (black) and suppresses the ocular drift present in the input (red)."*

### F4 — Sampling recipe  (`F4_sampling_recipe`)
- **Shows:** (a) CC vs the reverse start step t\* (t\*=1000 ⇒ full conditional generation), EOG and EMG;
  (b) ablation bars on single-channel EOG: ε → +eval-mode → +x0 → +full-generation (ours) vs SimpleCNN.
- **Takeaway:** the method only works with **full conditional generation** (EOG CC 0.59→0.90 as t\*→1000; a
  low-t\* SDEdit warm-start re-injects the artifact) and **x0-prediction** (0.85→0.90); reaching CNN parity.
- **Placement:** Experiments, method-analysis / ablation subsection (or end of Method). Text width.
- **Draft caption:** *"What makes conditional diffusion denoising work. (a) CC vs the reverse-process start
  step t\*: full conditional generation (t\*=1000) is best; an SDEdit-style warm-start re-injects the artifact.
  (b) Single-channel EOG ablation: x0-prediction and full generation bring SADDPM to parity with SimpleCNN."*

### F5 — Multi-channel denoising + artifact topography  (`F5_multichannel_{EOG,EMG}`)
- **Shows:** the artifact's spatial topography as a scalp topomap (frontal-dominant EOG / lateral EMG), then
  contaminated / denoised / clean as channel×time images for a 22-channel window.
- **Takeaway:** denoising all channels jointly exploits the artifact's spatial structure; the contaminated
  band is removed and the denoised image matches the clean one (multi-channel CC 0.99).
- **Placement:** Experiments, multi-channel subsection. Also doubles as the data/task illustration. Text width.
- **Draft caption:** *"Multi-channel joint denoising. Left: the (frontal-dominant) spatial topography of the
  injected ocular artifact. Right: contaminated, SADDPM-denoised, and clean 22-channel windows; the
  spatially-structured artifact is removed and the denoised output matches the ground truth."*

### F3 — Subject embedding-swap heatmap  (`F3_subject_swap_heatmap`)  ⭐ CENTERPIECE
- **Shows:** denoise subject *i*'s data using subject *j*'s embedding → CC matrix, for two regimes: shared
  artifact (left) and subject-specific artifact (right); diagonal = correct subject, off-diagonal = wrong.
- **Takeaway:** with a **shared** artifact the embedding is inert (uniform map, correct 0.988 ≈ wrong 0.987);
  with **subject-specific** artifacts the correct embedding is essential — a bright diagonal (0.993) against a
  teal background of wrong embeddings (0.830). One panel proves the subject mechanism is load-bearing.
- **Placement:** Experiments, the subject-aware subsection — the climax figure. Full text width (two panels).
- **Draft caption:** *"Is the subject embedding load-bearing? Each cell is the denoising CC when subject $i$'s
  data is denoised with subject $j$'s embedding (diagonal = correct subject). With a shared artifact (left)
  the embedding is irrelevant; with subject-specific artifacts (right) only the correct subject's embedding
  recovers the signal (diagonal 0.993 vs off-diagonal 0.830), showing the subject conditioning is load-bearing
  precisely when artifacts are subject-specific."*

### F6 — Power spectral density  (`F6_psd_{EOG,EMG}`)
- **Shows:** Welch PSD of contaminated / denoised / clean.
- **Takeaway:** the excess low-frequency ocular power in the input is removed and the denoised spectrum matches
  the clean one — the spectral complement to F1, supporting RRMSE_spectral.
- **Placement:** Experiments, beside F1 (qualitative) **or** Appendix if space is tight. Half/text width.
- **Draft caption:** *"Spectral recovery: the SADDPM-denoised PSD (blue) matches the clean spectrum (black),
  removing the excess low-frequency power introduced by the ocular artifact (red)."*

---

## Notes for assembly
- **EOG in main text, EMG in appendix.** F1/F2/F5/F6 each have an EMG twin (`*_EMG`). To save space, feature
  EOG in the main figures and move the EMG versions to an appendix, or stack EOG/EMG as two rows of one figure.
  Be honest about EMG in text: single-channel EMG is harder (our CC ~0.68 vs SimpleCNN 0.74) — F2_EMG shows this.
- **Figure budget.** Six new figures + the architecture schematic is on the high side for `acmsmall`. A lean
  main-text set is **F2, F1, F4, F5, F3** (5), with **F6 + all EMG twins** in an appendix.
- **Consistent style** across all figures: serif font, clean(black)/contaminated(red)/denoised(blue) color code,
  PDF vector output — already applied by `scripts/figgen_plot.py`.
- These figures correspond 1:1 to the figures requested in `MANUSCRIPT_UPDATES.md` §1.
