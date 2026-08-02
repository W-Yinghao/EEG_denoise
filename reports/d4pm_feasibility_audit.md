# D4PM feasibility audit

## Scope and disposition

This is a static audit only. No D4PM Python, MATLAB, Slurm job, data scan, or
download was run. The inspected checkout is the upstream repository at commit
`5be2b3c72973fea6c879e63cd83067ff66aace13` (2025-07-19).

**Disposition:** the as-released full EEGdenoiseNet D4PM protocol is not ready
to submit. The registered data lacks a verified ECG component, and the released
code has two execution-breaking inconsistencies described below. A repaired
EOG/EMG-only implementation is technically feasible, but it would be a
separately labelled reimplementation rather than an exact native reproduction.
D4PM remains the second-priority diffusion benchmark after EEGDfus.

## What D4PM is, and is not

D4PM is materially different from current CGDR/M2. It trains two conditional
diffusion component models: one predicts the clean EEG component and the other
the artifact component. At inference, both reverse chains are conditioned on
the observed mixture and an EOG/EMG/ECG class label, then coupled by a per-step
quadratic residual/data-consistency update for `y = x + gamma*h`. The released
example uses `lambda_dc=0.5`, `gamma=1`, and `eta=0.3`.

It does not use a population/context projector, participant calibration, or an
M2 final-hard-Q update. Therefore the negative current-CGDR/M2 result neither
tests nor predicts D4PM. Conversely, D4PM's artificial single-channel mixture
task cannot establish natural multichannel EEG restoration or personalized
operator value.

## Required data and current availability

The native data preparation unconditionally loads four 512-sample libraries:

| component | source expectation | released comment | current registered evidence |
|---|---|---:|---|
| clean EEG | `EEG_all_epochs.npy` | 4,514 epochs | verified `[4514,512]` |
| EOG | `EOG_all_epochs.npy` | 3,400 epochs | verified `[3400,512]` |
| EMG | `EMG_all_epochs.npy` | 5,598 epochs | verified `[5598,512]` |
| ECG | `ECG_all_epochs.npy` | 3,600 epochs | **not present in the current EEGdenoiseNet registry** |

The registered dataset is
`/projects/EEG-foundation-model/eegdenoisenet/github-8d290661146c7189c98cc04812d37371d4b9426c`.
Its registry records only EEG/EOG/EMG, at 256 Hz and 512 samples. A bounded
exact-name locator ended with `incomplete_timeout_no_absence_claim`, so this
audit does not claim that ECG is absent from the entire shared data root. It
does establish that ECG is not yet a verified input to a D4PM run. The native
loader cannot proceed without it.

The repository also contains an SSED preparation script expecting
`ssed_noise.npy` and `ssed_eeg.npy`, reshaping 15,162 x 400 samples to 512. It
does not have a corresponding D4PM training or evaluation entry point in this
commit, so the SSED script alone is not an executable second official protocol.

## Released protocol and budget

- One initial independent permutation is applied to clean EEG and each
  artifact library; there is no upstream seed.
- Each artifact type is split approximately 80/10/10 by row, followed by 11
  additional clean/artifact pairing permutations.
- Training SNR labels are sampled uniformly from -5 to +5 dB; test mixtures use
  11 levels over the same range.
- Each clean or artifact diffusion branch is configured for 4,000 epochs,
  batch size 1,024, Adam at `1e-3`, FP32, and 500 linear diffusion steps from
  beta `1e-4` to `0.02`.
- From the released array counts, the combined training set has 110,858 rows;
  `drop_last=True` gives 108 optimizer updates per epoch, or 432,000 updates per
  branch and 864,000 across the two branches. This is a large full-scale job,
  not a smoke test.
- Validation occurs every 10 epochs. The checkpoint contains model weights
  only; optimizer, scheduler, epoch, and RNG state are not saved, so the native
  trainer has no deterministic wall-time resume.
- The released evaluation script handles EOG only and evaluates only the first
  50 rows at each of 11 SNR levels (550 outputs), despite constructing 41,140
  EOG test mixtures. It is an example subset, not a complete evaluation.

## Independence and estimand limitations

The native split is not source-independent across artifact classes. EOG and
ECG both take overlapping prefixes of the same permuted EEG library but use
different split boundaries. The EMG route prepends the first 1,084 clean EEG
epochs and then appends all 4,514 clean epochs, duplicating clean sources before
splitting. Consequently a clean source epoch can occur under different
artifact labels and different train/validation/test roles. The 11 mixture
permutations further repeat source epochs within each split.

EEGdenoiseNet exposes no participant/session IDs here. Even a corrected
source-epoch split may claim only source-epoch separation, never participant
independence. Native results must be labelled artificial paired
EEG/EOG/EMG/ECG mixture results; they are not real-EEG clinical evidence.

The helper named `get_rms` returns mean-square rather than root-mean-square,
and the scaling coefficient uses that value with a square-root SNR factor. A
native reproduction must retain and disclose this exact mixture recipe; a
physically corrected SNR recipe is a separate reanalysis.

## Source defects that block an as-released run

1. **Artifact trainer NameError and checkpoint architecture mismatch.**
   `train_d4pm_artifacts.py` imports only
   `DualBranchDenoisingModel_noise` but instantiates the undefined name
   `DualBranchDenoisingModel`, so the released trainer stops with `NameError`.
   Merely importing that second class would not close the protocol: the noise
   class uses four attention heads and the ordinary class uses two, while
   `test_joint.py` constructs the four-head noise architecture to load the
   artifact checkpoint. A two-head checkpoint would therefore have incompatible
   attention parameter shapes.
2. **Prepared-output path mismatch.**
   Data preparation writes to hidden paths beginning `.data/data_for_*` and
   does not create those directories. The evaluation script reads
   `./data/data_for_test`. Even if hidden output directories were pre-created,
   the evaluator would look in a different path.

Additional reproducibility limitations are the absent seed, absent dependency
file, minimal README, no full-test runner, and model-only checkpoints. The
frozen repository also contains no license file; local scientific use and any
redistribution/vendor plan should be treated separately.

## Static `icml` compatibility

The frozen source imports only PyTorch, NumPy, SciPy, scikit-learn, pandas,
PyYAML, and the Python standard library. Existing Slurm environment evidence
for `/home/infres/yinwang/anaconda3/envs/icml` records Python 3.9.25,
PyTorch 2.8.0+cu128, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2,
pandas 2.3.3, and PyYAML 6.0.3, with CUDA available on L40S. Thus there is no
static missing-import blocker. This is not runtime validation; the released
architecture/checkpoint mismatch remains fatal, and batch 1,024 still requires
a GPU memory smoke test.

## Minimal scientifically honest implementation path

1. Do not submit the native full run until `ECG_all_epochs.npy` has a verified
   legal source and exact target path. Do not infer its absence from the timed
   out locator and do not substitute synthetic ECG silently.
2. Preserve an **as-released static/native-failure** record. Do not call a
   repaired execution exact official reproduction.
3. Implement a thin external adapter, without vendoring upstream source, that
   fixes paths and consistently uses one declared artifact architecture. The
   apparent intended repair is the imported four-head noise model, but that
   choice must be explicitly labelled `minimally_repaired_source-faithful`.
4. Run two frozen analyses if the data becomes complete:
   - seeded native split/mixture semantics, retaining and reporting source
     overlap and the released SNR recipe;
   - strict source-epoch split before reuse, pairing, or mixing, with no source
     epoch shared across train/validation/test or across artifact classes.
5. Match the official 4,000 epochs, 500 diffusion steps, FP32 objective, and
   optimizer-update budget. If batch 1,024 does not fit, use frozen gradient
   accumulation to preserve the effective batch and disclose that it is an
   execution adaptation.
6. Add resumable checkpoints containing both branch weights, optimizer,
   scheduler, epoch, RNG, and data-split state. Evaluate all frozen test rows
   for EOG, EMG, and ECG; keep the upstream 550-row EOG example as a separate
   diagnostic only.
7. Compare the dual diffusion method with a same-input, same-pairs,
   same-update-budget deterministic model. Report temporal/spectral RRMSE,
   correlation, SNR, latency, memory, and failure rate by artifact type and SNR.

Until those steps complete, D4PM status is **not tested / blocked on verified
ECG plus explicit source repair**. It supplies no evidence for or against the
current CGDR/M2 conclusion.
