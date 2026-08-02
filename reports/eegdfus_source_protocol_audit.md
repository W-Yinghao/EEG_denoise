# EEGDfus source and protocol audit

Status: `adapter_implemented_tests_not_run`

This HARNESS_LEVEL=1 audit is limited to the frozen external EEGDfus source,
the already registered EEGdenoiseNet arrays, and the local adapter. No Python
was run and no Slurm benchmark job was submitted during this implementation
step.

## Frozen source boundary

- Official repository: `https://github.com/XYH0118/EEGDfus.git`
- External checkout: `/home/infres/yinwang/denoiseNet/.external/EEGDfus`
- Frozen commit: `a19a652b3b6346188ae77067e1daf8b90cad005f`
- Checkout Slurm job: `919778`
- The frozen tree has no LICENSE/COPYING/NOTICE file. Consequently the adapter
  dynamically loads the official model, DDPM and metrics modules from the
  ignored checkout. It does not copy their source into this repository and
  makes no redistribution-license claim.

The registered data source is EEGdenoiseNet at
`/projects/EEG-foundation-model/eegdenoisenet/github-8d290661146c7189c98cc04812d37371d4b9426c`.
Its clean EEG, EOG and EMG arrays are single-channel source-epoch libraries.
The registered shapes are respectively `[4514,512]`, `[3400,512]`, and
`[5598,512]`. They contain no usable participant IDs; neither protocol may call
an epoch or split a participant.

## Upstream EOG/EMG protocol

The frozen `config/base.yaml`, `train_eegdnet.py`,
`Data_Preparation/data_prepare_eegdnet.py`, `DDPM.py`, and `utils.py` establish:

- 4000 epochs, batch size 512, Adam at 1e-3, StepLR every 1500 epochs, and
  gradient clipping at 1.0;
- a 500-step linear diffusion schedule from beta 1e-4 to 0.02;
- independent clean/artifact permutations, a 90/10 source partition, eleven
  additional permutations, train SNR sampled uniformly from -5 to 5 dB, and
  eleven evaluation levels from -5 to 5 dB;
- the native EOG branch truncates the permuted clean library to the smaller EOG
  library, while the native EMG branch prepends reused clean epochs to match
  the larger EMG library; the adapter reports this rather than relabelling the
  reused rows as independent sources;
- a row-level 80/20 train/validation split only after the eleven mixtures have
  been created. The same source epochs therefore occur in both native training
  and validation. This is retained and reported by `official_native`; it is
  not silently repaired;
- no seed in the EOG/EMG entry point or preparation code. The adapter registers
  seed 20260802 to make its wrapper reproducible while recording that this is
  an added execution constraint;
- upstream checkpoints contain model weights only, and the training entry
  point constructs but does not evaluate the full test set.

The upstream preparation function also writes test arrays under its own
`./data` tree. The adapter intentionally keeps those arrays in scheduled-job
memory and emits only aggregate metrics, leaving the ignored official checkout
read-only; this changes storage behavior, not the pairing or evaluation set.

The adapter preserves the 4000/512/500 and mixture semantics for `full`, adds
optimizer/scheduler/RNG checkpoint-resume, and evaluates every fixed SNR level.
Its tiny overrides are accepted only for the explicitly non-scientific
`smoke` stage.

The frozen `metrics.py` also has a spectral-metric incompatibility.  Its
`RRMSE_s` creates 400-bin PSD arrays but compares the clean PSD to
`zeros(clean.shape)`, which has 512 time samples.  Current sklearn rejects the
400-vs-512 output dimensions.  The adapter leaves official source untouched,
records the official spectral value as blocked, and reports only the explicitly
named `rrmse_spectral_corrected_psd_denominator_shape`, whose sole change is a
PSD-shaped zero denominator.  It is not labelled an exact official value.

## Strict source-epoch protocol

`strict_source_epoch` is separately labelled and is not presented as the
official native result. It independently freezes clean and artifact source
epochs into 72/18/10 training/validation/evaluation groups before pairing,
augmentation, or SNR mixing. Within each split, the smaller component library
is cycled only far enough to cover every source epoch in the larger library;
that reuse never crosses splits. The emitted split manifest uses
`source_epoch_not_participant` and rejects every cross-split source overlap.

## Matched deterministic arm

Each native/strict diffusion cell has one deterministic cell using the same
dynamically loaded `DualBranchDenoisingModel` backbone. The observed noisy
epoch is its only visible input and is supplied to both backbone streams with a
fixed scalar condition. It receives the exact same prepared pairs, split,
batch size, optimizer schedule, 4000 epochs and number of optimizer updates as
the paired diffusion cell. It has no diffusion latent or iterative sampling.

The frozen eight-task matrix is protocol × EOG/EMG ×
conditional-diffusion/matched-deterministic. This comparison is only a paired
single-channel EOG/EMG stress test, not participant-specific operator evidence.

## SSED issues retained as audit findings

The SSED path is not run or repaired in this adapter. The frozen source has the
following separate issues:

1. `train_ssed.py` creates `val_test_idx`, then splits
   `range(len(val_test_idx))` and applies those rebased positions directly to
   the full dataset. Validation/test can therefore select the wrong rows and
   overlap the training selection.
2. `data_prepare_ssed.py` returns clean EEG followed by constructed noisy EEG,
   while `train_ssed.py` assigns those to `X_train, y_train` and constructs
   `TensorDataset(y_train, X_train)`. The generic training loop consumes the
   first element as clean, reversing the intended clean/noisy roles.
3. A test loader is constructed but is not passed into the upstream training
   call or evaluated there.

These findings do not authorize an unlabelled correction of official-native
results.

## Prepared execution boundary

- Adapter: `src/eeg_cgdr/experiments/eegdfus_benchmark.py`
- Config: `configs/baselines/eegdfus_native_strict.yaml`
- Unit/semantic tests: `tests/unit/test_cgdr_eegdfus_benchmark.py`
- Intended CLI mode/stages: `eegdfus-benchmark cpu-tests|smoke|full`
- Intended CPU semantic-test profile: `cpu` without an array
- Intended smoke profiles: `V100-32GB`, `L40S`, `gpu-any`
- Intended full profiles: `gpu-any`, `A100`, `H100`
- Intended array: `0-7%8`
- Stable checkpoints:
  `/home/infres/yinwang/denoiseNet/artifacts/checkpoints/eegdfus_benchmark/...`
  (ignored and never committed)
- Small outputs:
  `/home/infres/yinwang/denoiseNet/results/cgdr/eegdfus_benchmark/...`

No benchmark result exists yet. Smoke output must be labelled
`completed_tiny_smoke_only`; only a 4000-epoch, 500-step `full` task is eligible
for the stated single-channel stress-test scope.
