# Minimal implementation plan

Status: frozen for audit-only work; scientific implementation is blocked by attachment conflicts.

## Scope decision

The local repository is an existing SADDPM/BCI-IV-2a worktree, not the empty project assumed by the bootstrap narrative. Its package, configs, scripts, tests, checkpoints/results, and manuscript claims predate the current CSPD population-posterior contract. They are preserved for provenance but are not accepted as verified inputs or results.

The first legal change set is limited to control-plane/audit infrastructure: frozen Slurm and environment mappings, a single `sbatch` submitter, scheduled attachment/runtime/data inventory tools, dataset registry schemas, provenance schemas, disabled configuration stubs, and engineering tests. No old scientific run is relabeled and no population base, P0, gate, backup fit, outer-test result, or paper value is produced while `CONFLICT-SCI-001` through `003` remain unresolved.

## Reuse

- Reuse the existing Git worktree and branch without reset, clean, mass formatting, or replacement.
- Reuse the registered `eeg2025` and `icml` environments exactly as audited; exchange only hashed files between jobs.
- Reuse general-purpose low-level utilities only after test-level review: deterministic seed helpers, atomic/checkpoint utilities, tensor-shape helpers, and diffusion schedule primitives may be wrapped rather than copied.
- Reuse no old empirical result, checkpoint, participant embedding, subject-conditional claim, hard-coded split, V100 submission directive, login-node command, W&B identity, or home-directory data cache as evidence for the current protocol.

## Extend

- Extend `scripts/slurm/` with the audited central submitter and job-specific payloads that always `cd` to the fixed code root, select one registered environment, capture allocation/config hashes, and atomically write status.
- Extend `.gitignore` to keep EEG data, partial downloads/extractions, secrets, weights, large arrays, Slurm streams, caches, and temporary review data out of Git.
- Extend repository reporting with bootstrap, attachment, runtime, inventory, decision, conflict, and final-delivery ledgers.
- Extend testing only through Slurm jobs. Existing tests remain historical until run against a frozen environment and classified for relevance.

## Add after audit evidence

- `datasets/registry/`, `manifests/`, `schemas/`, and `splits/` containing metadata only; no raw EEG is copied into the worktree.
- A new contract-focused package namespace (`src/eeg_cspd/`) only where the old `saddpm` API cannot satisfy the leakage and semantic boundary. It will depend on data/splits → population base → support operator/mask → energy correction → sampler/baselines → gates → evaluation, never in reverse.
- Immutable support/query types, visible-field audit, dataset-specific prior registry, run/context/gate status schemas, and failure-retaining aggregation.
- Required configuration families with `TBD-PREREG` thresholds and B1–B6 `enabled: false`.
- Synthetic fixtures only for interface, math, leakage, endpoint, failure, and provenance tests; never result tables or gate decisions.

## Explicitly not enabled

- No dataset is `verified_available` or `missing` until the full scheduled root inventory plus license/source/sample-read audit completes.
- No download, unpack, preprocessing, derived-data publication, training, sampling, outer evaluation, or plotting is authorized by this plan alone.
- No unconditional prior may stand in for `NULL`; no old subject-ID-conditioned SADDPM route may stand in for `PopulationPosteriorBase`.
- No B1–B6 implementation may fit/search/compare. Stubs remain disabled even if old manuscript language could be read as opening them after two gates.
- No push, release, remote archive, data upload, environment install/upgrade, or paper editing is planned.

## Immediate ordered work

1. Complete and record the repository audit against the attachment/user contract.
2. Run the full fixed-root read-only inventory through `cpu-high`, preserving bounded-I/O and
   pre-termination `PARTIAL` evidence. The Phase-I walker has no unsafe cursor-resume claim: if
   the five-day audited allocation is still insufficient, retain that attempt and submit a new
   complete walk rather than treating a non-snapshot cursor as coverage proof.
3. From the admitted inventory evidence, implement and schedule the version/license/access,
   full-integrity and bounded sample-read jobs specified in
   `reports/targeted_dataset_audit_plan.md`; do not publish preliminary registry records from
   Phase-I path/name hits alone.
4. After all four targeted attempts have immutable terminal evidence, run registry schema plus
   semantic validation through `cpu` and publish one evidence-bounded record per candidate by
   no-replace operation.
5. Add and run administrative schema/safety tests through `cpu`; fix only the new scoped files.
6. Stop before population-base semantics. Resume scientific implementation only after the joint manuscript/server authority resolves the population-mask/NULL, real-EEG gate-order, and backup-order conflicts and Stage A has a verified real EEG path plus frozen splits.

## Concurrent-change policy

Before every submission, capture HEAD, branch, remote, tracked-diff hash, untracked-path hash, config hash, and input manifests. If another task changes a consumed file, mark the run `stale_input`; do not combine versions. Stage only files created or intentionally edited for this contract in any local commit, and never include unrelated user changes.
