# Related-work re-audit (cutoff 2026-08-06)

This is an exploration-round evidence map, not a novelty claim. Searches cover
IEEE/publisher pages, PubMed/PMC, arXiv, Crossref-indexed DOI pages, author
repositories, and official project documentation. Google Scholar counts are
not used: the bounded Scholar-domain query did not return an auditable Scholar
result surface, so no CAPTCHA bypass or citation-count inference was attempted.

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

- A July 2026 deep-learning EEG-denoising review (DOI
  10.1088/1741-2552/ae89e6) organizes the field around target construction,
  representation, architecture, objectives, and evaluation, and explicitly
  identifies selective/multi-task denoising and downstream validation as
  deployment issues. It is useful scope evidence, but not evidence that a
  disjoint-support subject-conditioned restorer already exists.

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
- **RestoreGrad** (ICML 2025) learns an observation-informed prior jointly
  with a conditional diffusion restorer. It is relevant to the question of
  whether a Gaussian starting prior discards observation information, but it
  does not provide EEG-specific disjoint-support subject calibration.
- **MultiDiffNet** (NeuroAI Multimodal Intelligence at AAAI 2026) reports
  subject/session-disjoint EEG *decoding* across SSVEP, MI, P300, and imagined
  speech. Its diffusion latent is optimized for classification rather than
  artifact waveform restoration, so it strengthens the decoding side of the
  taxonomy without testing the candidate denoising gap.
- **DiffEEG** (arXiv:2607.11578) uses denoising-diffusion pretraining for
  generic EEG representations followed by seizure classification. Its strict
  patient-wise evidence is relevant to representation generalization, but it
  neither estimates an artifact-removal output nor uses disjoint subject
  calibration support.

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
- **CT-DCENet** (IEEE JBHI 2025, DOI 10.1109/JBHI.2025.3535592) is a
  deterministic CNN--Transformer dual-stage ensemble on public paired
  artifact benchmarks. It is relevant as a strong population comparator, not
  as disjoint-support personalization.
- **BandRouteNet** (arXiv:2604.24428) is a 2026 population artifact-removal
  preprint that adaptively routes corrections across frequency and time. Its
  selective correction is relevant to P-C, but its reported EEGdenoiseNet
  evidence does not identify unseen-subject support effects.
- **ReHA-Net** (Scientific Reports 2026) combines ReVIN, multiscale
  convolution, and attention for population EEG artifact removal. It provides
  a timely normalization/statistics control for P-D; it does not expose an
  early-support to later-query unseen-participant protocol in the audited
  publication surface.

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
- HyperEEGNet uses resting-state target-user EEG to generate EEGNet weights for
  motor-imagery classification under leave-subject-out evaluation. Its support
  mechanism motivates raw-support-to-parameter mappings, but the reported
  nine-participant classification setting is not an artifact-restoration
  validation. TSMNet uses domain-specific SPD batch-normalization statistics,
  and the 2025 Euclidean Alignment review distinguishes efficient target-domain
  alignment from personalized waveform reconstruction.
- The names `PhysioPFM` and `TCPL` were included in the search vocabulary.
  TCPL could be resolved to a classification prompt-learning paper; no unique,
  primary artifact-restoration paper or official implementation named
  `PhysioPFM` was verifiable from the queried publisher/index sources by the
  cutoff. It is therefore recorded as unresolved rather than inferred from a
  similarly named project.
- Conditional diffusion has also been used for ambulatory ECG noise reduction
  (DOI 10.1063/5.0222123). It supports transfer of conditional score
  restoration ideas across physiology, but it does not establish EEG
  subject-calibration or neural preservation.

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
- CT-DCENet: https://doi.org/10.1109/JBHI.2025.3535592
- BandRouteNet: https://arxiv.org/abs/2604.24428
- Ambulatory ECG conditional diffusion: https://doi.org/10.1063/5.0222123
- RestoreGrad: https://proceedings.mlr.press/v267/lee25ai.html
- HyperEEGNet: https://openreview.net/forum?id=04RGjODVj3
- TSMNet: https://openreview.net/forum?id=pp7onaiM4VB
- Euclidean Alignment review: https://doi.org/10.1088/1741-2552/addd49
- 2026 deep-learning EEG-denoising review: https://doi.org/10.1088/1741-2552/ae89e6
- ReHA-Net: https://www.nature.com/articles/s41598-025-28855-0
- MultiDiffNet: https://proceedings.mlr.press/v308/zhang26a.html
- DiffEEG: https://arxiv.org/abs/2607.11578
