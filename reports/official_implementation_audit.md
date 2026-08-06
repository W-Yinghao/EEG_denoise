# Official implementation audit

J0 completed on Slurm jobs 924788 and 924795. All nine named author
repositories were available at the commits below. A successful checkout is
only source availability; it is not numerical reproduction.

| Method | Commit | Source/runtime audit | v3 status |
|---|---|---|---|
| EEGDfus | `a19a652` | Official code has `train_eegdnet.py`, `train_ssed.py`, conditional DDPM code, and data preparation. Native EEGdenoiseNet validation is a post-mixing random split and has no participant identity. | Official-source port; the already completed strict source-epoch rerun is reusable as a population diffusion baseline, not subject evidence. |
| D4PM | `5be2b3c` | Official code exposes clean/artifact branches and joint sampling. `test_joint.py` passes the true per-mixture artifact label to inference. | Deployable reconstruction must remove that label; the label-conditioned run is extra-information diagnostic only. |
| Essentia | `f4ee52f` | Official repository contains the model and contrastive training code, but `TrainContrastive.py` leaves dataset paths/loaders as `None` placeholders and internal code retains author-local paths. | `reconstructed_from_paper`; not exact official reproduction. |
| EEGOAR-Net | `4e67cb9` | Official TensorFlow architecture, pretrained `h5` weights, channel standardization helpers, and an example with three subjects are present; no official training/split entry is included. | Official pretrained inference can be ported after TensorFlow/channel compatibility checks; frozen-split retraining is unavailable from the repository. |
| DS-DDPM | `12c339a` | Minimal research code contains subject-ID embeddings/classification loss and author-specific experiment machinery. | Audit/background only: known closed-set subject labels and brain-dynamics generation differ from disjoint-support artifact restoration. |
| SGEYESUB | `2c95b4f` | MATLAB reference implementation with calibration and evaluation demo is present. | Current Python implementation remains `source-faithful port` until MATLAB parity; never relabelled exact reproduction. |
| DeepSeparator | `7dca7dc` | Official PyTorch 1.9/Python 3.6 training/prediction code targets EEGdenoiseNet single-channel mixtures. | Ported population artifact-removal baseline; no subject-awareness claim. |
| EEGANet | `587f690` | Official repository documents subject-independent tests and SGE data, but provides only a TensorFlow-era implementation surface. | Calibration-free population comparator after compatibility validation. |
| MobileBCI data code | `c103740` | Official MATLAB loaders/evaluators cite OSF R7S9B, CC-BY-4.0, 24 participants, scalp/ear EEG, four EOG and IMU sensors. The official OSF acquisition and bounded header index completed. All 198 `sourcedata` channel tables retain 46 EEG plus four EOG. The processed BIDS EEG tables omit EOG and have 46 EEG plus 27 IMU for 196/198 records; two records lack those processed IMU entries. | Available for a future development route with ses-02 standing support and ses-03/04/05 motion queries; no signal outcomes were opened by this audit. |

Additional source tracing located the authors' ART repository
(`CNElab-Plus/ArtifactRemovalTransformer`) and the IC-U-Net PyTorch release
(`roseDwayane/AIEEG`). ART constructs supervised multichannel targets from
ICA-labelled brain/non-brain source mixtures; IC-U-Net ships an inference
surface and a fixed 30-channel example. Neither source was silently treated as
an exact frozen-split baseline: their target construction and channel contract
are materially different from the Klados/SGE support protocol. They remain
audited candidate ports, while the full v3 science matrix retains an
information-matched deterministic comparator and the strong project POP.

The existing complete EEGDfus reconstruction in the main worktree used eight
full cells (official/strict-source-epoch × EOG/EMG × diffusion/deterministic),
including 208k--344k optimizer updates per cell. It is retained as prior
population-baseline evidence rather than rerun merely to duplicate compute.
Its official spectral metric has an upstream denominator-shape defect; only
the separately labelled corrected PSD metric is usable. Numerical parity,
physical-scale validity, and frozen-split results remain separate evidence
from repository availability.
