# SGEYESUB MATLAB parity preflight

Status: `official_checkout_ready; blocked_matlab_executable_unavailable`

This is a targeted HARNESS_LEVEL=1 preflight and authorized follow-up. It did
not scan the shared EEG root, compute hashes, open EEG signals, or run
MATLAB/Python on the login node. The official public code was later downloaded
only by the bounded CPU Slurm job documented below.

## Existing evidence checked

- Dataset registry: `datasets/registry/sgeyesub.json` records the OSF release at
  `/projects/EEG-foundation-model/sgeyesub/osf-2qgrd` as available.
- Reference audit: `reports/sgeyesub_reference_audit.md` freezes the official
  repository at commit `2c95b4f46f37670d25399ac0fdd705ae18248b25` and keeps
  the Python port explicitly unvalidated against MATLAB.
- Structure audit: `reports/sgeyesub_structure_audit.md` supports only the
  release-internal block-1 support to block-2 query protocol. It does not
  establish a paper EEGDS mapping.
- Existing protocol metadata identifies representative compatible evaluation
  stems `study02_p02`, `study04_p03`, and `study05_p06` and preserves exact
  study/layout/reference/sampling cells.

## Original checkout observation

The one-level SGEYESUB data directory contains only `osf-2qgrd`. Targeted
checks of the known/proposed code locations found no local official checkout:

- `/projects/EEG-foundation-model/sgeyesub/eyeartifactcorrection`
- `/projects/EEG-foundation-model/eyeartifactcorrection`
- `vendor/eyeartifactcorrection`
- `external/eyeartifactcorrection`
- `/home/infres/yinwang/eyeartifactcorrection`

This was not an assertion about every path on the server. No broad filesystem
search was performed. At that point this preflight had no download
authorization, so no checkout was created. The later explicit authorization
is recorded separately below; code remains in the ignored code-root directory
rather than the EEG data root.

## Authorized official checkout

CPU Slurm job `919777` ran the bounded checkout command
`scripts/slurm/submit.sh cpu sgeyesub_reference_checkout`. Its payload cloned
`https://github.com/rkobler/eyeartifactcorrection.git` without a worktree,
checked out the pre-registered commit, verified the expected origin and key
official files, then atomically published the clean detached checkout at:

`/home/infres/yinwang/denoiseNet/.external/eyeartifactcorrection`

The expected and actual commit are both
`2c95b4f46f37670d25399ac0fdd705ae18248b25`. The machine-readable result is
`reports/sgeyesub_reference_checkout/919777/checkout.json` with status
`verified_frozen_checkout`. Existing or concurrently appearing targets are
read-only checked and never fetched, checked out, or overwritten. The ignored
`.external/` checkout is not intended for Git publication.

## Prepared execution boundary

- `scripts/slurm/jobs/sgeyesub_matlab_probe.sbatch` performs a CPU-node-only
  `command -v`, targeted module lookup/load, and MATLAB license/runtime probe.
  Raw module and license diagnostics are discarded so license-server details
  cannot enter logs.
- `scripts/matlab/sgeyesub_parity_runner.m` accepts one frozen fixture and the
  official checkout. It verifies the official commit, exact channel order and
  `type == EEG` indices, fits block 1 only, applies to block 2 only, and rejects
  query annotations/outcomes.
- `configs/cgdr/sgeyesub_matlab_parity.yaml` freezes the representative-first
  order (`study02`, `study04`, `study05`) and then expansion to every stem that
  passes the same per-record compatibility checks.

If MATLAB or its license is unavailable, official cross-runtime parity remains
`blocked`; the Python output retains the label
`source_faithful_not_numerically_cross_validated`, and other non-MATLAB work may
continue.

## CPU probe result

Slurm CPU job `919775` completed the bounded probe. The compute node exposed a
module command, but its targeted MATLAB lookup found no MATLAB module; loading
`matlab`/`MATLAB` failed, `command -v matlab` remained empty, and therefore no
license command was run. The sanitized machine-readable result is
`reports/sgeyesub_matlab_probe/919775/availability.json`.

Submission command: `scripts/slurm/submit.sh cpu sgeyesub_matlab_probe`.

The official checkout prerequisite is now satisfied. The MATLAB parity route
remains blocked only on a usable MATLAB executable/runtime. This does not block
the existing Python release-internal implementation or other deterministic
analyses, but that implementation must retain its non-cross-validated label
until cross-runtime parity is actually run.
