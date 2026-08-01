# Data lookup status

Slurm job `918918` was cancelled at `2026-08-01T11:04:22Z` after the user
clarified that `/projects/EEG-foundation-model` is a private 30 TB workspace and
does not need an industrial, exhaustive inventory.

The scanner closed cleanly as `PARTIAL` with exit code `4`. It had visited
2,423,427 entries, hashed 10.28 GB across 216,671 allowlisted metadata files,
and emitted 2.23 GB of JSON. Its start/end Git, environment, mount and input
guards matched, so the cancellation was not `stale_input`. These partial files
are cancellation evidence only; no absence, availability, integrity, license,
or scientific claim may be derived from them.

The replacement workflow is deliberately small:

1. List the data-root top level and run one bounded-depth, name-only Slurm
   locator for the four requested datasets.
2. If a name matches, inspect only that candidate directory.
3. If no name matches, verify the official source and access terms, then use a
   versioned `.partial` download under the data root.
4. Record one small registry entry per dataset with path, version, source,
   access state and a short readability note. Use official checksums when they
   exist; do not hash the whole dataset merely to create local provenance.

The exhaustive scanner must not be resubmitted.
