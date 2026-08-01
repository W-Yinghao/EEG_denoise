# Phase-II implementation readiness audit

Observed at: `2026-08-01T06:47:17Z`

Status: **design ready / executable implementation deferred until Phase-I terminal evidence**

This is a login-node, read-only compatibility audit of small repository text and Slurm control
state. It did not run Python, tests, imports, source APIs, parsers, hashes over the data root, or
any data access. It does not assert that a dataset exists or authorize a download.

## Live Phase-I boundary

Inventory job `918918` remains `PENDING (Priority)` on `cpu-high`, with no allocation, runtime,
scan artifact, or exit evidence. The scheduler estimate `2026-08-05T00:40:00` is not completion
evidence. The submission request and the current protected inputs still agree:

| Guard | Frozen/current SHA-256 |
|---|---|
| cluster config | `2b1ffc570ada77717bf3baa008d4820d1060bc81cb7724b23edb7795612f2275` |
| environment config | `ab41490c0328840d042ed1aa61f2c4c1cd623996a6b713dcdf4cb0dcd7166eb4` |
| inventory job script | `73f96a23be0c0e014070ffb85ef5048981e3ad3f5109d4371e484fe96693a015` |
| submitter | `1dd330f215093848ad9de9a36845fa762c22e3f3732334c721651072e1374100` |
| complete `scripts/contract/*.py` bundle | `d267d01c96ec2af9b15875dd94654152b40b76a3c8ba945a57b1ce82a9ad2215` |
| complete `scripts/slurm/jobs/*.sbatch` bundle | `0f1f338781494a809257b72cd4550d0b99623b013c3e837c620c8a1fab24bc48` |
| immutable submission request | `e975168283697be00c5e35a2b577203263a2f2f9f611b9efce2354042a0404ce` |

The hashes were recomputed using the same absolute-path bundle algorithm as the submitter. A
relative-path reproduction gives a different digest because the bundle format includes path text
and must not be compared to the frozen value.

Adding, deleting, or editing any contract helper or registered job payload before 918918 starts
would change a whole-directory bundle and cause startup exit `75`. Changing either config, the
submitter, inventory payload, or request would do the same. Once scanning starts, the scanner also
compares start/end Git HEAD, branch, remotes, tracked diff, all untracked paths, scoped untracked
content, and `eeg2025` package state. Therefore the whole observed worktree—not only the scanner
file—must remain unchanged from allocation start until its atomic terminal state. A concurrent
change is retained as `stale_input`; it is never repaired by silently accepting mixed evidence.

## Existing implementation versus required contract

The repository currently has no `targeted_dataset_audit` or `validate_dataset_registry` helper,
Slurm payload, source-freeze manifest set, semantic policy, registry generator, or corresponding
contract tests. The current registry schema validates only a weak outer JSON shape.

| Area | Current evidence | Required Phase-II addition |
|---|---|---|
| Dataset ID | arbitrary nonempty string | exact four-ID enum, filename/ID agreement, one record each |
| Dataset status | seven values are enumerated | evidence-prerequisite decision table; job status remains separate |
| Phase-I binding | job ID only | job/attempt/state, terminal/status/manifest/shard hashes and one publication snapshot |
| Source identity | one nullable string | structured, revision-bound data/code identities with no cross-source substitution |
| License/access | free-form string | separate data/code license, scope, response hash, effective permission and DUA evidence |
| Integrity | free-form strategy and counts | authenticated expected set, observed full manifest, coverage and official/local hash semantics |
| Sample parsing | free-form string | outcome enum, parser/version, sample-plan hash, strata, byte cap and limited scope |
| EEG fields | unconstrained objects | per-field evidence state/scope/hash and explicit unknown/conflict |
| Paths | arbitrary strings | unique normalized dataset roots below fixed data root, safe ancestry and sensitive-component rejection |
| Review time | nullable unformatted string | required RFC3339 UTC publication review time |
| Publication | no set schema | exactly four records, atomic/no-replace set semantics and shared Phase-I snapshot |

The first implementation batch, after 918918 terminates, must include at least:

- source-freeze request manifests for the four candidate IDs;
- schemas for targeted status, targeted artifact manifests, strengthened registry records and the
  four-record publication set;
- one structured registry semantic policy and deterministic seven-state decision implementation;
- targeted audit, registry generation and registry validation helpers;
- `cpu-high` targeted and `cpu` test/validator Slurm payloads;
- synthetic contract fixtures and unit/integration/leakage/smoke tests;
- ignored private targeted job directories with wording that still forbids raw participant/session
  paths, raw signal and credentials; and
- either audited `afterany` support or the preregistered controller-waits-for-all-terminals path.

If the integrity estimate requires an array, the same first bundle must validate array/task IDs,
deterministic nonoverlapping shard plans, aggregation and terminal semantics. It is not legal to
truncate the manifest because the existing request validator assumes a non-array job.

Legacy `configs/data.yaml`, `scripts/link_local_dataset.py`, `environment.yml`, top-level legacy
tests and old MOABB/V100 result paths are not reusable evidence for this implementation.

## Runtime authority rollover

Adding the Phase-II helpers and jobs changes the globally audited contract and job bundles. Jobs
`918908` and `918909` then remain valid only for the historical Phase-I chain; they cannot authorize
the new payloads even if the Conda package lists are unchanged.

The required prospective order is:

1. Admit the immutable terminal package from 918918. A valid COMPLETE may route all four audits; a
   valid PARTIAL can route only bounded positive-hit preservation and never supports `missing`.
   Exit `75`, missing atomic state or unsafe hashes opens no candidate data.
2. Implement and commit the entire Phase-II code/config/schema/test/job bundle once. Include the
   registry validator before targeted audits start so no code is added between their start and
   registry publication.
3. Transition the two-environment registry through its registered pending state. Submit fresh
   `eeg2025` CPU and `icml` L40S strict audits, without package installation or upgrade; register
   their evidence and pass verified-state control validation.
4. Run the new administrative schema, semantic, path-safety, leakage and smoke tests through the
   `cpu` submit path. A code fix that changes an audited bundle requires another authority cycle.
5. Submit four independent `cpu-high` targeted audits through the fresh `eeg2025` authority. Each
   request binds the same Phase-I snapshot and a candidate-specific sanitized request; paths,
   participant IDs and credentials never appear in argv.
6. Wait for and hash-bind every targeted terminal package. Scheduler `afterany` is only a wake-up
   condition, not evidence admission. If dependency history is no longer scheduler-resolvable,
   submit only after the controller observes all terminal packages; never silently weaken a failed
   dependency.
7. Run the `cpu` registry validator and publish exactly four records by no-replace only if all
   schema, semantic, provenance, privacy and publication-set checks pass.

Targeted requests keep both `phase1_runtime_audit_job_id=918908` and the future
`target_runtime_audit_job_id`; the latter is execution authority and the former is immutable parent
provenance. The Phase-I ID is never rewritten to the new audit ID.

## Source-freeze boundary

Only routing expectations already supported by local review can be preregistered. API response
hashes, exact source-check UTC timestamps, credentials, effective permissions, selected manifests,
local paths, local bytes, parser results and closure remain `TBD-after-authorized-audit`.

- **Klados-Bamidis:** freeze entity `wb6yvr725d`, DOI `10.17632/wb6yvr725d.1` and requested v1.
  Never borrow v4 members, hashes or dictionary evidence. The CC BY 4.0 value remains a source-page
  claim until response/scope evidence is captured. Phase II never enables 60/120 s cells.
- **SGEYESUB:** keep OSF data project `2qgrd` and GitHub reference code as distinct identities.
  The current 403 is a probe failure, not evidence of `restricted` or `missing`; LGPL is code-only.
  A commit prefix is not final code provenance, and native SGEYESUB never reads an oracle span.
- **Eye-BCI:** freeze one Synapse project-tree enumeration and each FileEntity's version, handle,
  size and content digest. Project/Folder IDs have no invented version and etag is not a checksum.
  Paper/project CC0 and public wording do not prove entity-level access or local completeness.
- **EEGdenoiseNet:** separate GitHub code, selected GIN tree, historical annex-add commit and
  license-provenance commit. Annex pointers are not payloads. Standard members use the 256 Hz
  expectation; only admitted `_512hz` EEG/EMG members use 512 Hz, and no 512 Hz EOG is assumed.

For every source, version, license and access are independent verdicts. Every member, size,
checksum and license response binds one selected source revision. A complete local SHA-256
manifest identifies a local copy but cannot by itself prove official equivalence.

## Fail-closed verification matrix

The Slurm test job must reject, at minimum:

- duplicate JSON keys, extra fields, invalid IDs/timestamps/hashes and filename/ID mismatch;
- missing/changed terminal markers, cross-attempt evidence, PARTIAL/no-hit to `missing`, and sample
  success to `verified_available`;
- relative/root-escaping/symlink/TOCTOU paths and persisted participant/session identifiers;
- hit counts used as full counts, official expected size used as observed bytes, annex stubs used
  as content, and unsupported parsers mislabeled `corrupt`;
- v4 evidence used for Klados v1, OSF data license inferred from LGPL, Synapse etag used as content
  hash, and GitHub/GIN/historical commits substituted for one another;
- tokens, cookies, signed URLs, raw FileEntity handles where sensitive, signal samples or decoded
  arrays written to reports;
- fewer or more than four registry records, mixed Phase-I snapshots or source revisions, partial
  publication, and overwrite of an existing registry; and
- login-node execution, wrong environment/profile/allocation, bundle/lock drift, or non-atomic
  replaceable output.

Positive fixtures must cover all seven legal dataset states, shared Phase-I snapshot enforcement,
and bounded registry drafting from a structurally complete legitimate non-success audit terminal.
The audited environment lists `jsonschema`, but its import and Draft 2020-12 behavior still require
a Slurm smoke; no dependency installation or upgrade is authorized.

## Science stop boundary

This readiness audit changes none of the Stage A, population-base, P0 or G1-G5 statuses. There is
still no registry, verified real EEG path, split, prior, checkpoint, NULL equivalence test or outer
fold. The three scientific authority conflicts remain unresolved, and B1-B6 remain disabled.
