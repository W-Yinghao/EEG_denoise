# Lightweight implementation plan

Status: targeted data acquisition complete; dataset mapping and confirmatory science remain blocked.

## Active scope

This is a private research repository maintained as one coherent project. The
active workflow deliberately does **not** use a full 30 TB inventory, per-file
hash ledger, CAS publication, bundle authority rollover or repeated environment
audit. Git history, small dataset records and Slurm job/status files are enough
for engineering provenance. Prior configs likewise use readable split,
checkpoint and Git references rather than local SHA-256 fields.

The safeguards that still matter are unchanged: run project work through
Slurm, keep EEG bytes in `/projects/EEG-foundation-model`, do not expose query
targets or test identities to fitting, keep `NULL` observation-conditioned, and
respect the population-base → P0 → G1–G5 order. B1–B6 remain disabled.

## Completed data path

- Exhaustive inventory `918918` was cancelled and must not be rerun.
- Bounded basename lookup `919129` inspected 100,000 names in 10.9 seconds,
  found no target name and read no file content.
- EEGdenoiseNet's existing official 256 Hz payload was copied to the data root
  by `919131`, header-checked, and registered. Legacy code paths now point to it
  through links created by `919148`.
- Eye-BCI is registered as restricted because Synapse requires a user token.
- Klados v1's official archive was downloaded by `919153`. A bounded RAR4
  header diagnostic (`919182`) saw the candidate names `Contaminated_Data.mat`,
  `HEOG.mat` and `Pure_Data.mat`, but its listing was incomplete and it did not
  extract or read MAT data. No more custom archive parsing is planned.
- SGEYESUB is public CC BY 4.0 and 1.50 GiB. After correcting unstable default
  API pagination, job `919172` published all 178 files and `919175` read one
  finite EEGLAB epoch from each study.
- Native SGEYESUB is frozen separately to the official `2c95b4f` reference:
  fixed horizontal, vertical and residual-blink directions, `alpha=1`,
  `beta=0.01`, and no oracle/projector interpretation. Final metadata job
  `919218` matched all 59 participant SET/FDT/block-metadata stems without
  opening FDT data. It found six layouts and only blocks 1/2 in every delivered
  SET, so the paper-level three-block mapping remains explicitly unresolved.
- Lightweight submitter/self-test `919190`, minimal Klados size/signature audit
  `919191`, and final lightweight gate/prior config self-test `919220` passed.

## Code reuse

Reuse:

- `saddpm/models/unet1d.py` only in its non-subject-conditioned form;
- diffusion schedule and numerical sampling primitives from
  `saddpm/diffusion/gaussian_diffusion.py`;
- the EEGdenoiseNet component loader, now pointed at the data root;
- MNE/SciPy/HDF5 for targeted sample reads after download.

Do not reuse as the new population base:

- `ConditionalDiffusionDenoiser`, because it hides `y` in a black-box network
  and cannot expose separate clean-prior score and `E0`;
- subject embeddings, old M3–M12 checkpoints, old outer results or epoch-random
  splits as evidence for the new participant-level protocol.

## Minimal population-base design

After Stage A and scientific conflict resolution, add only:

1. immutable query/context and support/query split types;
2. an unconditional clean-prior score component trained on legal clean targets;
3. an explicit population observation energy
   `E0(x;y)=0.5*(x-y)^T Lambda0 (x-y)`, with `Lambda0` fitted only on outer
   training data and initially labelled generalized Bayes;
4. a `PopulationPosteriorBase` that combines those two components using the
   same query `y` at every step;
5. a `NullSampler` that directly delegates to `base.sample` and cannot accept
   operator, mask, attenuation or correction arguments;
6. unit/leakage tests for `y` dependence, same-seed NULL equality, zero
   individualized calls, PSD/endpoints and forbidden input fields.

EEGdenoiseNet may exercise this path as an engineering paired stress test, but
its current files have no participant/session grouping and cannot by themselves
close Stage B. Klados becomes the first formal candidate only if extraction
confirms clean components and participant structure. SGEYESUB remains
evaluation-only unless a compatible clean prior is separately justified.

## Remaining blockers and next order

1. Resolve the official SGEYESUB study-to-paper mapping and the observed
   two-block versus paper three-block discrepancy before freezing a native
   split; do not infer it from participant counts alone.
2. Leave Klados at `present_unverified` unless an already-approved RAR reader
   becomes available; do not mutate shared Conda environments for this step.
3. Identify a legally usable clean-target source for a formal population prior;
   SGEYESUB is evaluation-only and EEGdenoiseNet remains engineering-only.
4. Resolve `CONFLICT-SCI-001` by explicitly separating population `E0` from
   individualized `a_tau`/mask calls. Until then the base above is a design, not
   a confirmatory implementation.
5. Only then implement and test population base/NULL, followed by P0 and the
   gates in order.

No remote push, release, dataset upload or paper-result substitution is
authorized by this plan.
