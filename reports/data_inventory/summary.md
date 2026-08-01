# Data inventory scheduler status

Generated at `2026-08-01T06:26:51Z`.

Slurm job `918918` was submitted at `2026-08-01T06:06:16Z` and is currently
`PENDING` for reason `Priority`. The scheduler currently estimates a start at
`2026-08-05T00:40:00 Europe/Paris`; this is an estimate only, not a reservation,
allocation, or completion commitment.

The submitted request uses profile/partition `cpu-high`, 8 CPUs, 64 GiB memory,
and a five-day wall-clock limit. Its `afterok:918908` dependency is satisfied.
The job has not received an allocation and has produced no inventory output,
exit status, or scan evidence.

## Scientific status

No dataset-state claim is authorized while the job remains pending. In
particular, no state may yet be generated or inferred for
`klados_bamidis_v1`, `sgeyesub`, `eye_bci`, or `eegdenoisenet`, and no dataset
registry record may be created from this pending scheduler record.

Even if the Phase-I scan later finishes as `COMPLETE`, Phase-I path/name
evidence alone cannot establish either `verified_available` or `missing`. A hit
can support only `present_unverified`; a complete scan with no hit remains
`unknown`. Targeted version, license/access, and sample-read audits are required
before creating the evidence-backed registry records.

## Next action

When the allocation starts, freeze the Git/worktree and scoped untracked input
state for the duration of the scan. After completion, verify the immutable job
status, scan status/marker, manifest hash, coverage, mount evidence, and every
listed output shard before using any Phase-I evidence. Then run the targeted
version, license/access, and sample-read audits before writing dataset registry
records.
