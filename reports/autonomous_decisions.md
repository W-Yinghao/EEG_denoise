# Autonomous decisions

## 2026-08-01: preserve non-empty local repository

- Evidence: local branch `master` at `2f408f3cdc4895347b6a567b913159b1c60a0b50`, nine tracked modifications, and many untracked files; remote had no refs.
- Decision: preserve all local content, add only the specified empty remote as `origin`, and extend in place. Do not clone over, reinitialize, reset, clean, or push.
- Alternatives rejected: replacing the worktree or treating existing results as validated. Either would violate preservation/provenance requirements.
- Impact: new contract files remain separable from pre-existing work, while old claims/results stay untrusted until audited.

## 2026-08-01: supplemental PDF extraction with `icml` on CPU

- Evidence: scheduled attachment job `918740` found that `eeg2025` lacks Poppler utilities and `pypdf`; its audited lock does contain `pdfminer.six` but no renderer. The audited `icml` lock contains PyMuPDF 1.26.5.
- Decision: submit a short, CPU-profile Slurm job using the already registered `icml` environment solely to extract text, annotations, links, metadata, and 110-DPI page renderings. No package or environment changes are made.
- Alternatives: text-only extraction with `pdfminer.six` would leave visual/formula/table inspection unresolved; requesting an environment change is unnecessary because an existing registered environment contains the required read-only library.
- Impact: this is an administrative attachment-review exception, not a scientific CPU/GPU comparison and not evidence for any model result. The environment lock and allocation remain explicit.

## 2026-08-01: bounded full-root inventory walltime and interruption semantics

- Evidence: the audited `cpu-high` partition permits five days; the data root is a 30 TiB NFS4
  mount, and a one-day request cannot be assumed sufficient before the file count is known.
- Decision: request the audited five-day maximum and a batch-shell `USR1` signal ten minutes
  before timeout. The batch wrapper forwards that signal to the fixed `eeg2025` Python process,
  which closes atomic shards and publishes `PARTIAL` evidence. Phase I does not claim a
  resumable, consistent filesystem cursor because the live NFS namespace is not a snapshot.
- Alternatives rejected: an unbounded scan, silently labeling a timeout complete, or resuming
  from an unstable directory-order cursor. If the allocation is insufficient, a later attempt
  must be a fresh full walk while preserving the original job and failure evidence.
- Impact: at most one single-process read-only walk runs, metadata hashing and output remain
  budgeted, and `COMPLETE` means only that no coverage error was observed—not dataset integrity,
  absence, license, or sample readability.

## 2026-08-01: control-plane validation submission rejected before `sbatch`

- Evidence: the first hardened `scripts/slurm/submit.sh cpu validate_control_plane` invocation
  exited `2` locally before calling `sbatch`; Bash rejected an inline constraint regular expression
  containing `&`. No Slurm job ID was created and no payload ran.
- Decision: retain the failed attempt in this ledger, move the registered constraint expression to
  a quoted shell variable, and retry the same administrative validation without changing its
  profile, resources, or validation scope.
- Impact: this is a control-plane parser defect, not a job or scientific failure. No result may cite
  the rejected invocation as a Slurm test.
