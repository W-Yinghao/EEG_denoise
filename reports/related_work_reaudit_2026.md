# Related-work re-audit (cutoff 2026-08-06)

This is an exploration-round evidence map, not a novelty claim. Searches cover
IEEE/publisher pages, PubMed/PMC, arXiv, Crossref-indexed DOI pages, author
repositories, and official project documentation. Google Scholar counts are
not used because they are not needed for the scientific taxonomy and may be
blocked by CAPTCHA.

## Taxonomy

| Restoration object | Individual condition | Adaptation mechanism | Typical evidence | Relevance to v3 |
|---|---|---|---|---|
| Full noisy EEG / single-channel waveform | Noisy query or artifact class | Conditional DDPM / posterior sampling | Paired synthetic or semi-synthetic clean targets | Strong population diffusion baselines, not subject-aware evidence |
| Clean and artifact distributions | Artifact type and mixture assumptions | Dual priors / joint posterior | EEGdenoiseNet/SSED-style paired mixtures | D4PM tests diffusion design, not unseen-participant calibration |
| Subject-domain representation | Known training subject/domain label | Domain decomposition / subject classifier | Cross-subject decoding | DS-DDPM is classification/domain invariance, not physiological artifact restoration |
| Naturalistic EEG | Query-side pose or motion sensors | Sensor-conditioned diffusion | Mobile recordings with side information | Pose-informed diffusion has extra query-time information absent from our deployable setting |
| EEG artifact correction | EOG/ICA/reference calibration | Regression, projection, ICA/ASR | Natural EEG and preservation diagnostics | Required non-diffusion controls |
| Unseen-subject task decoding | Few labeled/resting target samples | prompts, adapters, meta-learning, DSBN/EA | Classification accuracy | Mechanism inspiration for P-A/P-B/P-D, not direct denoising evidence |
| EEG artifact correction | Disjoint calibration support, no query external sensor | raw-support tokens, support-only adapter, selective use | paired mechanism + natural EEG | Candidate gap; "first" is not asserted |

## EEG diffusion restoration

- **EEGDfus** (IEEE JBHI, DOI 10.1109/JBHI.2024.3504716) is an EEG
  denoising diffusion method with an official repository
  (`XYH0118/EEGDfus`). Its native role is a paired EEG artifact-removal
  benchmark. The repository and split semantics are audited separately before
  any result is called a reproduction.
- **D4PM** (arXiv:2509.14302; `flysnow1024/D4PM`) models clean EEG and artifact
  distributions with dual branches and joint posterior sampling. It directly
  bears on diffusion utility, but its artifact-type/SNR information must be
  separated into deployable and oracle-information evaluations.
- **Essentia** (ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10887905) uses
  semantic guidance for EEG artifact removal. Code completeness and the
  relationship between semantic labels and deployable query information are
  audited before reconstruction.
- **Pose-informed EEG diffusion** (bioRxiv DOI
  10.1101/2023.12.08.567146) conditions naturalistic EEG denoising on pose.
  This is query-side auxiliary sensing, not query-disjoint support
  calibration.
- **DS-DDPM** (arXiv:2305.04200; `duanyiqun/DS-DDPM`) decomposes subject-domain
  variance for brain-dynamics recognition and supervises subject variance with
  a subject classifier. Its task is cross-domain recognition rather than
  ocular/motion artifact waveform restoration.

## Artifact-removal baselines

- **EEGOAR-Net** (DOI 10.1016/j.bspc.2025.108147;
  `dmarcos97/EEGOAR-Net`) is treated as a strong calibration-free ocular
  removal baseline subject to official-code audit.
- **SGEYESUB** (NeuroImage DOI 10.1016/j.neuroimage.2020.117000;
  `rkobler/eyeartifactcorrection`) explicitly estimates corneo-retinal and
  eyelid-related corrections. Its native MATLAB/source semantics remain
  separate from the source-faithful Python port.
- **DeepSeparator** (`ncclabsustech/DeepSeparator`, arXiv:2112.00989) learns a
  latent separation model on EEGdenoiseNet-style data and reports both
  semi-synthetic and real task-related evaluations.
- **IC-U-Net** (arXiv:2111.10026) uses ICA-derived brain/non-brain source
  mixtures to train a U-Net. Its target provenance differs from paired clean
  EEG and must remain explicit.
- **ART** (arXiv:2409.07326), ICA+ICLabel, ASR, and MNE EOGRegression cover
  transformer, component-removal, burst-reconstruction, and linear EOG
  regression baselines. MNE documents EOGRegression as fitting artifact-channel
  regression coefficients; EEGLAB documents ASR as identifying and
  reconstructing high-variance subspaces relative to clean calibration.

## Subject adaptation mechanisms

- TCPL uses few-shot task-conditioned prompt tokens under meta-learning for
  motor-imagery classification. Its support is labeled task calibration and
  its endpoint is classification, not denoising.
- ResTL (arXiv:2405.19346; `SionAn/MICCAI2024-ResTL`) adapts cross-subject MI
  classifiers using target resting-state EEG. This motivates independent raw
  support encoders while not establishing artifact-removal efficacy.
- TSMNet/DSBN and Euclidean-alignment families demonstrate inexpensive
  distribution/statistic adaptation for EEG classification. P-D uses one such
  statistic control to test whether a complex support model adds value.
- LoRA/adapters and prompt/meta-learning are mechanism candidates only. P-B is
  explicitly a direct support-only upper bound before any amortized
  hypernetwork is allowed.

## Candidate novelty boundary

The defensible candidate gap at this stage is:

> support-conditioned personalized diffusion for physiological EEG artifact
> removal on unseen participants under disjoint early-support to later-query
> evaluation.

The audit does **not** yet establish that this is the first such method. A
positive v3 screen would still require backward/forward citation tracking and
independent confirmation before a novelty statement.

## Primary links

- EEGDfus: https://doi.org/10.1109/JBHI.2024.3504716 ; https://github.com/XYH0118/EEGDfus
- D4PM: https://arxiv.org/abs/2509.14302 ; https://github.com/flysnow1024/D4PM
- DS-DDPM: https://arxiv.org/abs/2305.04200 ; https://github.com/duanyiqun/DS-DDPM
- Essentia: https://doi.org/10.1109/ICASSP49660.2025.10887905 ; https://github.com/NKU-EmbeddedSystem/Essentia
- Pose-informed diffusion: https://doi.org/10.1101/2023.12.08.567146
- EEGOAR-Net: https://doi.org/10.1016/j.bspc.2025.108147 ; https://github.com/dmarcos97/EEGOAR-Net
- SGEYESUB: https://doi.org/10.1016/j.neuroimage.2020.117000 ; https://github.com/rkobler/eyeartifactcorrection
- DeepSeparator: https://arxiv.org/abs/2112.00989 ; https://github.com/ncclabsustech/DeepSeparator
- IC-U-Net: https://arxiv.org/abs/2111.10026
- ART: https://arxiv.org/abs/2409.07326
- MNE EOGRegression: https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html
- ICLabel Python API: https://mne.tools/mne-icalabel/stable/api/index.html
- EEGLAB ASR: https://eeglab.org/tutorials/06_RejectArtifacts/cleanrawdata.html
- TCPL: https://doi.org/10.3389/fnins.2025.1689286
- ResTL: https://arxiv.org/abs/2405.19346
