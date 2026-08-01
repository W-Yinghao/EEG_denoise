# Phase-II targeted dataset audit contract

## Purpose and authority boundary

This contract defines the read-only Slurm work that may begin only after inventory job
`918918` reaches a terminal state. Its purpose is to turn weak Phase-I path/name hints into
dataset-specific evidence about release identity, access and license, local completeness,
integrity, parsability, fields, channels, sampling, participants, and sessions. It does not
authorize training, preprocessing, scientific evaluation, data download, archive extraction,
or publication of a dataset claim.

Phase-I candidate matches are routing hints, not trusted dataset roots. A filename substring,
README mention, code checkout, archive name, broken symlink, or cache entry must not be treated
as a dataset. Likewise, a successful bounded sample parse is not evidence that all release
members are present or intact.

The implementation should add two Slurm payloads after this plan is approved:

1. `targeted_dataset_audit`: source/version/access audit, candidate-root resolution, complete
   local release inventory, integrity comparison, and bounded header/sample parsing for one
   dataset ID.
2. `validate_dataset_registry`: schema plus semantic validation and no-replace publication of
   the four small registry JSON records.

Both payloads must be invoked only through `scripts/slurm/submit.sh`. This document does not
itself authorize adding those payloads or changing code, configuration, or schema.

## Terminal Phase-I admission

The login-node controller may read the small terminal status and submit work, but all validation
and data access occur inside Slurm.

### COMPLETE admission

For an `afterok:918918` submission, the audit job must require all of the following before it
opens a candidate path:

- `reports/data_inventory/jobs/918918/attempt-0/job_status.json` says `completed`, exit code 0,
  scanner exit code 0, profile `cpu-high`, and runtime audit job `918908`.
- `scan/status.json`, `scan/COMPLETE`, and `scan/process_exit_code.txt` agree on `COMPLETE` and
  exit code 0, with an empty `partial_reasons` list.
- The SHA-256 of `scan/inventory_manifest.json` equals the value pinned by `scan/status.json`.
- Manifest job, attempt, fixed code root, fixed data root, runtime audit, guards, and dynamic
  start/end state agree. Every guard is `matched`, and there is no stale-input record.
- Every entry/error shard named by the manifest is a regular no-symlink file below the exact
  attempt directory and matches its recorded byte count, record count, and SHA-256.
- `coverage.json`, `mounts.json`, and `candidates.json` are captured by hash in the Phase-II
  request. Coverage is `COMPLETE`, mount signatures agree, and candidate IDs are unique and
  exactly the four preregistered IDs.

Only this path can later support a dataset-level `missing` decision, and even then only after the
candidate-specific exhaustive locator described below also has an authoritative marker set.
Phase-I no-hit by itself still means `unknown`.

### PARTIAL admission and preservation

If 918918 exits nonzero but publishes a structurally valid `PARTIAL` result, a controller may
submit a separate `afternotok:918918` targeted evidence-preservation job. It must pin the PARTIAL
manifest hash and all available shard hashes exactly as above, retain every partial reason and
coverage boundary, and never splice its evidence into a later COMPLETE inventory.

- A retained candidate hit may be audited at the exact observed path, because it is positive
  presence evidence, but it is not evidence that the copy is unique or that the remainder of the
  data root was covered.
- A no-hit under PARTIAL remains `unknown`; it can never become `missing`.
- A fully anchored release located by a positive hit may undergo its own complete release audit,
  but the global data-root inventory remains incomplete and Stage A remains blocked.
- Exit 75 `stale_input`, missing atomic exit evidence, unsafe output ancestry, manifest/hash
  mismatch, or a preflight failure makes Phase-I evidence inadmissible. Preserve it as failure
  evidence and emit a blocked Phase-II status without opening candidate data.

The controller must wait for the terminal files before constructing either submission because the
expected manifest and output hashes are not known in advance. It must not submit both paths with
placeholder hashes.

## Common frozen inputs and guards

Each dataset audit request must contain and hash the following:

- `dataset_id`, restricted to `klados_bamidis_v1`, `sgeyesub`, `eye_bci`, or
  `eegdenoisenet`;
- fixed `data_root=/projects/EEG-foundation-model` and fixed
  `code_root=/home/infres/yinwang/denoiseNet`;
- Phase-I job/attempt/state, terminal status hashes, manifest hash, coverage/mount/candidate
  hashes, and every shard hash used by the targeted locator;
- the exact official source anchors and a source-metadata request manifest that records release
  or entity identifiers but contains no cookie, token, signed URL, or credential;
- the registered `eeg2025` environment path and two distinct runtime references: Phase-I audit
  job `918908` plus its explicit/pip locks as historical inventory provenance, and the fresh
  Phase-II `eeg2025` strict-audit job, locks, compatibility status, and bundle hashes established
  after the targeted implementation is frozen;
- Git remote, branch, commit, tracked-diff hash, relevant untracked-content hash, job-script hash,
  submitter hash, and contract-bundle hash;
- proposed dataset-level candidate roots as untrusted Phase-I evidence, including whether evidence
  was truncated and whether each hit represented a directory, symlink, archive, cache, partial
  path, code file, or metadata file. A normalized absolute release root may be persisted only
  after safe-ancestry and sensitive-component checks; participant/session-bearing member hits are
  represented by keyed path IDs and are never copied verbatim into the request;
- a deterministic, frozen file-list partition for complete integrity work and a separate frozen
  deterministic sample plan.

The payload must run in the registered `eeg2025` environment on `cpu-high`; it must not fall back
to `icml`, another environment, another partition, or login-node Python. It must verify actual
partition, CPUs, memory, node, dependency, and environment before access.

### Runtime-authority rollover

Job `918918` was submitted with immutable contract and Slurm-job bundle hashes. Adding any
Phase-II helper under `scripts/contract/`, any payload under `scripts/slurm/jobs/`, or changing
the submitter/configuration before that job reaches a terminal state would invalidate its startup
guards. Therefore no Phase-II executable, schema, configuration, manifest, or test implementation
is added while 918918 is pending or running.

After the admitted Phase-I terminal evidence is frozen, implement the complete Phase-II bundle as
one prospective version. That bundle change makes the 918908/918909 runtime audits historical;
it does not imply either Conda environment changed. Transition the environment registry through
its audited pending state, rerun the strict `eeg2025` CPU and `icml` L40S audits without installing
or upgrading anything, register their new immutable evidence, and run verified-state control
validation. The targeted audit request must use the new `eeg2025` audit as its execution authority
while retaining 918908 only as the Phase-I parent reference. No targeted data access begins until
the new control validation and the Phase-II administrative tests pass.

All dataset access is read-only and descriptor-relative. Use `lstat`, `O_NOFOLLOW`,
`O_CLOEXEC`, and `O_RDONLY`; request `O_NOATIME` where supported and record whether fallback may
have changed access time. Normalize every path and require it to remain below the fixed data root.
Never follow an observed Phase-I symlink. A symlink target may be considered only as a separately
guarded path if its resolved target is still inside the fixed data root, its complete ancestry is
audited, and the source manifest explicitly admits the layout. Any escape, path replacement,
device change not explained by the frozen mount map, or file identity change fails closed.

Official API and repository metadata requests are also Slurm payload work. Credentials, when
legally available, must arrive through an approved secret channel and be absent from argv,
environment dumps, URLs, stdout, stderr, and reports. Source responses retained in reports must
be stripped of signed URLs and identity-bearing access metadata before hashing and publication.

### Source-freeze record

Each candidate gets an independent source-freeze record; `version`, `license`, and `access` are
separate verdicts and an unknown value in one field is never filled from another field or from a
paper's general availability statement. At minimum the record contains:

- identity: `dataset_id`, source kind, canonical landing URL, canonical metadata/download API,
  DOI, publisher/owner, and source-check UTC time;
- revision: requested version, source-declared version, selected revision, revision type
  (`release`, full commit, entity version, or mutable project), and whether the source is mutable;
- licensing: data-license name/URL and exact evidence URL/hash; any reference-code license is a
  separate field and never substitutes for a data license;
- access: access class, metadata/download authentication requirements, DUA/terms, access-
  requirement IDs and actions, whether an approved credential is available, and the probe job ID;
- members: authoritative manifest source, relative path, source file/entity ID and version, byte
  size, content type, and official checksum algorithm/value;
- observed closure: local/downloaded SHA-256, safe archive-member manifest hash when applicable,
  manifest and sample-read verdicts, and the scientifically allowed role.

Unknown or unavailable fields use explicit `unknown`/`TBD-after-authorized-audit` values. They are
not omitted, guessed, copied from a later release, or completed from an unauthenticated mirror.

## Resource and I/O estimate

The preflight must write a resource estimate before submitting integrity work. It derives
candidate file count and bytes from the admitted Phase-I shards; it must not use a guessed dataset
size.

- Profile: `cpu-high`, 8 CPUs, 64 GiB, no GPU.
- Enumeration estimate: admitted candidate entries divided by the audited ceiling of 200 lstat
  operations/second, with a 1.5 safety factor.
- Integrity estimate: bytes selected for full hashing divided by 32 MiB/second, with a 1.5 safety
  factor. The estimate must separately list official-checksum verification bytes, locally
  generated SHA-256 bytes, metadata-only bytes, and bytes not legally or technically readable.
- Sample parsing estimate: selected sample count, maximum header/read window, parser timeout, and
  maximum in-memory allocation. Header parsers must not load a whole recording merely to report
  shape and metadata.
- Output estimate: JSONL record count and bytes; raw EEG, decoded arrays, and participant IDs are
  forbidden in the code root.
- Wall time: round the sum upward to an hour, subject to the audited five-day `cpu-high` maximum.
  If the estimate cannot fit, partition the sorted frozen file manifest into deterministic
  no-overlap job-array shards and aggregate only with `afterok` dependencies. Do not silently omit
  files to fit the wall clock.

Hashing should use one bounded sequential reader per filesystem by default. More workers require
an audited filesystem I/O limit. A preemption or wall-time signal closes the current shard,
publishes PARTIAL state, and permits only a new run ID over the frozen list of unfinished files;
completed no-replace shards remain immutable and are referenced by hash.

## Candidate-root resolution

The locator reads the complete Phase-I entry shards rather than trusting `candidates.json`
alone. For each candidate it builds a minimal ancestor set of exact matching paths, removes
descendant amplification, separates code, metadata, archive, cache, hidden, partial, and symlink
evidence, then compares the remaining layout with official release markers obtained from the
source anchor.

No path is promoted to a dataset root until at least two independent facts agree, for example an
official release/entity identifier plus an official member name, or a release manifest plus a
matching file checksum. A directory name and a README mention are not independent facts. Archive
members are not inferred from the archive filename; archive inspection, if later needed, uses a
separate safe read-only member-list audit and does not extract.

For COMPLETE Phase-I with no weak hit, a targeted locator may rescan the full fixed root only for
a frozen allowlist of authoritative release markers and content signatures. It may declare
`missing` only when that candidate-specific search covers every admitted mount and the official
marker set is sufficient to recognize the release independent of directory naming. Otherwise the
result remains `unknown`.

## Full integrity inventory versus bounded sample parsing

These are separate artifacts and separate verdicts.

### Full local release integrity

Once a release root is anchored, enumerate every regular member without following symlinks. Save
a pseudonymized relative-path identifier, kind, bytes, mtime, device/inode identity, readability,
expected source member ID, expected size/checksum when available, and observed checksum. Compare
the complete local member set with the official versioned release manifest. Raw participant-bearing
path components are used only in process memory: persisted evidence replaces them with stable
research pseudonyms or keyed path IDs, and any approved mapping is kept outside the code root and
outside job logs.

- An official checksum is used only when its algorithm, member identity, and release version are
  authenticated by the source metadata.
- If no official checksum exists, a complete local SHA-256 manifest can identify the local copy
  and detect future drift, but it cannot by itself prove that the copy matches the official
  release.
- File count and byte count in a registry come only from this anchored complete member list, never
  from Phase-I candidate hit counts.
- Pointer files, git-annex placeholders, HTML error pages, archive stubs, zero-length unexpected
  files, missing sidecars, and `.partial` paths are not valid data members.
- A local member list that is internally hash-complete but cannot be compared with authoritative
  release identity remains `present_unverified` unless other official completeness evidence is
  sufficient.

### Bounded parsability and field audit

After the complete member list is frozen, group files by actual discovered format, release role,
and session/layout stratum. Select deterministic samples by sorted SHA-256 of normalized relative
paths: first, median, and last when a stratum has at least three members, otherwise every member in
that stratum. Cap only the parser reads, not the full integrity inventory; record any cap and all
unrepresented strata. Parser output contains header/shape/type/channel/reference/sampling/event
metadata only, never signal samples.

Every parser is disabled by default. It is enabled only after both the source manifest and file
signature admit the discovered format:

- `.mat`: inspect the MATLAB signature/version first; use a bounded SciPy header reader for
  classic MAT or an HDF5 metadata reader for v7.3 only when that package is verified in
  `eeg2025`.
- `.set`: require an EEGLAB-compatible MAT/header signature. If it references `.fdt`, validate
  the exact no-escape relative sidecar, dtype/order metadata, and size equation before a bounded
  reader opens it.
- `.fdt`: never parse standalone; admit it only through a validated `.set` reference.
- `.cnt`: use a registered CNT reader only after source metadata and header signature identify the
  discovered CNT variant.
- `.npy`: require the NPY magic/version, read the header with `allow_pickle=False`, and use
  read-only mmap only if shape inspection requires it.
- Any other discovered suffix remains unsupported until its signature, source role, bounded
  reader, import evidence, and failure behavior are added to the frozen allowlist.

A successful sample parse proves only that the selected file/header was readable by that exact
parser. It does not prove release completeness, integrity of unsampled files, clean-target
semantics, participant independence, usable support duration, or scientific suitability. An
unsupported format is `not_attempted`, not `corrupt`. A parse failure supports `corrupt` only when
the file is an authenticated expected member, its bytes are fully present, and the failure is a
structural/data error rather than missing permission, absent dependency, or unsupported variant.

## Dataset-specific audit plans

### Klados-Bamidis version 1

Source identity is fixed to Mendeley dataset `wb6yvr725d`, DOI
`10.17632/wb6yvr725d.1`, version 1, not a later Mendeley version. Capture the v1
DOI/landing metadata, publication/update time, official v1 member list, member
sizes/checksums if supplied, data dictionary, and data license. A version 4 API response or
manifest must not be borrowed to validate version 1. A current landing page or candidate
directory name does not prove that local files are version 1.

After discovery, enable a `.mat` parser only for actual `.mat` members admitted by the v1 source
manifest and MATLAB signature. Record variable names, shapes, numeric types, sampling-rate
metadata, channel count/order/reference evidence, EEG/EOG roles, contaminated/clean/component
roles, participant/record identifiers under research pseudonyms, and any generation/mixing
metadata. Do not interpret an arbitrary matrix as EEG or a column as a physiological source.

For each unique native recording reached by the complete release manifest, compute native duration
only as original sample count divided by the audited native sampling rate. Record content hashes
and source member identity to detect duplicates. Resampling, interpolation, repeated epochs,
concatenating the same block, overlapping windows, or sliding-window reuse must not increase
independent support duration. Phase-II never enables a 60/120 s configuration cell. It may emit
`evidence_ready` only after an exhaustive metadata/header/context/overlap audit covers every
potential native block and proves that distinct, same-context, query-disjoint, non-overlapping
blocks provide the duration. The bounded first/median/last sample plan cannot establish this; when
exhaustive block evidence is absent, emit `N/A` or `unknown`. This is availability evidence, not a
gate or model result.

### SGEYESUB data and native reference code

Treat the OSF project `2qgrd` data release and the native reference code as two separate inputs.
For OSF, capture project/component/file version IDs, official file tree, sizes/checksums, data
license, access status, and synchronization/data-dictionary evidence. OSF data version, license,
and access remain `unknown` until that audit succeeds. For the reference code, resolve the
official `rkobler/eyeartifactcorrection` repository and require the expected first-party commit
prefix `2c95b4f` to resolve to one unambiguous full commit hash. Freeze that full hash rather than
a moving branch, record the LGPL code license and exact license text/hash, and hash the native
command/config used. The prefix alone is not registry provenance. Code availability or LGPL code
license does not establish OSF data availability, version, access, or data license.

Only if actual admitted files are `.set`/`.fdt` may the EEGLAB pair checks run. Validate the
`.set` header, no-escape `.fdt` reference, channel/sample/epoch dimensions, dtype/order, sidecar
byte count, EEG/EOG/reference channels, event markers, sampling rate, and calibration/session
structure. Audit the native code/config for its calibration window, reference signals, rank
selection, projection/correction rules, and default preprocessing. Preserve those values as
native provenance; do not substitute oracle span or relabel the method as oracle. If the release
uses another discovered format, select only the corresponding admitted parser and report the
format actually observed.

### Eye-BCI

Resolve Synapse entity `syn64005218` through an approved account if required. Freeze a project-tree
snapshot hash and enumerate every admitted child `FileEntity`; for each one capture entity ID,
`versionNumber`, `fileHandleId`, `contentMd5`, `contentSize`, content type, and the redacted response
hash. Record Project/Folder identity and response hashes without inventing version numbers for
entities that do not expose them; an `etag` is concurrency metadata, not a content checksum.
Evaluate access requirements and the effective `READ` and `DOWNLOAD`
permissions for every required entity rather than inferring them from the project page. Capture
the official child manifest, data-use/access terms, and any required DUA. A project ID, public
paper, DOI, or project-level CC0 label does not prove that the underlying bytes are anonymous,
legally downloadable by the current account, complete, or readable. If current
credentials/approval are absent, emit `restricted` without attempting to bypass access or logging
credential material.

Do not assume CNT exists. If authenticated expected members and local signatures show a CNT
variant, use the admitted CNT parser; otherwise use only the parser allowlisted for the actual
discovered format. Inventory and cross-check EEG, EOG, eye-tracking/gaze, event/trigger, task,
montage, reference, channel order, sampling rate, synchronization, participant, session, and
paradigm fields. Record session/participant identities only as stable research pseudonyms. Verify
whether simultaneous streams and event clocks can be joined and identify missing/unsynchronized
streams, but bound every such conclusion to sessions covered by complete metadata or to the exact
sampled context. Bounded parsing cannot establish joinability or completeness for all 63 sessions.
Do not infer paired clean EEG merely because EEG and eye data coexist. The evidence must state
which real cross-session or cross-paradigm drift axes are actually evidenced.

### EEGdenoiseNet

Keep the official code repository commit, its code license, the GIN data repository commit, and
the dataset license/access evidence separate. Freeze and revalidate these independent first-party
identities rather than mapping one repository's commit onto the other:

- `reference_code_github`: `https://github.com/ncclabsustech/EEGdenoiseNet`, expected full commit
  `8d290661146c7189c98cc04812d37371d4b9426c`, with the MIT **code** license;
- `gin_dataset`: `https://gin.g-node.org/NCClab/EEGdenoiseNet`, selected mutable-tree snapshot
  `2bca0a94d1bd41dfa67358934d5b15d1efc0b73a`, with the GIN source-level CC0 **data** license
  evidence; and
- historical GIN commit `7d242bb7a1f0914df6dfe95703a1a20e55dcdfe6`, retained only as
  provenance for the commit that added the 512 Hz annex members, never as the selected current
  release/tree revision.

Before use, resolve each full hash at the named first-party repository and verify repository
identity; a moving branch, truncated prefix, cross-repository SHA, or historical data-directory
commit is not final release provenance. Resolve the exact selected GIN tree and git-annex
metadata; for every expected annex member record the annex key, expected byte size, and MD5/MD5E
digest when the authenticated key supplies them. The four first-party 512 Hz members with
published size/MD5 pairs must match those exact per-file values, but that four-file evidence must
not be generalized to any other member. Inspect the remainder of the tree and every pointer
independently. Detect pointer/placeholders whose annex content is absent, including
ordinary-looking archive members that contain only an annex stub. A Git clone or archive
containing annex pointer files is not a complete dataset.

Do not assume `.mat` or `.npy` files exist. Enable those readers only for actual source-admitted
members with valid signatures. For `.mat`, retain bounded variable/shape/type evidence; for
`.npy`, use `allow_pickle=False` and header/read-only mmap inspection. Compare clean EEG, EOG, and
EMG role labels, sample/epoch definition, sampling rate, source split metadata, and expected
member size/checksum with the official release evidence. The source-level expectation for all
listed epoch arrays is single-channel, two-second segments. Standard members are expected at
256 Hz, while only source-admitted EEG/EMG members whose names end in `_512hz` are expected at
512 Hz; the first-party tree does not provide an `_512hz` EOG member. Every local header must
confirm its own sampling rate and shape. Never apply the 512 Hz expectation to a standard member
or use that mismatch to declare corruption. Explicitly record the observed channel and sample
dimensions. Single-channel segments may support a generic EOG/EMG pressure test but cannot
establish an individual multichannel transfer operator, montage-specific projector, or
participant-level personalization claim.

## Immutable outputs and job state

Each attempt writes only below
`reports/data_inventory/targeted/<dataset_id>/jobs/<job_id>/attempt-<restart>/` with `umask 077`.
No output is written to the data root. The attempt contains:

```text
request.json
resolved_inputs.json
input_guards.json
slurm_submission.json
slurm_allocation.json
environment.lock
source_identity.json
license_access.json
candidate_resolution.json
integrity_plan.json
integrity/manifest.json
integrity/shards/*.jsonl
sample_plan.json
sample_results.jsonl
field_evidence.json
failures.jsonl
artifacts_manifest.json
status.json
COMPLETE | PARTIAL | FAILED | STALE_INPUT | BLOCKED
process_exit_code.txt
```

Write each artifact to a job-local `.partial`, `fsync`, then publish by same-filesystem hard link
or equivalent no-replace primitive. An existing final path, symlink component, changed input,
changed candidate identity, mount drift, or source-version drift fails closed. `status.json` is
written last and pins the SHA-256, bytes, schema, and record count of every artifact. The terminal
marker repeats the status hash. Never overwrite or merge attempts; recovery gets a new run ID and
links immutable parent/shard hashes.

Provenance records the source/entity/repository versions, official response hashes after secret
redaction, Phase-I manifest/shard hashes, code/Git/diff hashes, environment/audit/lock hashes,
actual Slurm allocation, pseudonymized path-resolution decisions, parser IDs and versions, read
byte limits, expected and observed checksums, all failures, and start/end guard hashes. Neither
private job evidence nor curated reports/registries may persist participant-identifying path
components or raw signal. Exact participant/session-bearing member paths may exist only
ephemerally in process memory; persisted member references use source entity IDs and stable
research pseudonyms/keyed path IDs. A dataset-level absolute release root may be persisted only
after normalization below the fixed data root, safe-ancestry validation, and a check that none of
its components encode a participant or session identity.

Job state and dataset state are separate. `COMPLETE` means the planned audit executed without an
unaccounted failure; it does not mean the dataset is `verified_available`. `PARTIAL`, `FAILED`,
`STALE_INPUT`, and `BLOCKED` are retained in the denominator and cannot be rewritten as successful
dataset evidence.

## Seven-state dataset decision table

Apply the first matching rule below and retain all secondary conditions in machine-readable
evidence:

| Dataset status | Required evidence |
|---|---|
| `restricted` | The authoritative source requires an account, approval, DUA, or license grant that is not currently available, so legal source/local validation cannot proceed. Do not inspect protected signal content. |
| `corrupt` | An authenticated expected member is fully present but fails its authoritative checksum, authenticated size, or an admitted structural parser; permission and unsupported-parser failures do not qualify. |
| `partial` | The release root is anchored, but an authoritative expected member/sidecar/annex object is absent or truncated, or an approved transaction remains `.partial`. Overall Phase-I `PARTIAL` alone is not a dataset `partial` decision. |
| `verified_available` | Exact release/entity version, data license, and legal access are verified; the anchored complete local member set matches authoritative completeness/integrity evidence; key formats pass the frozen bounded parser plan; and fields/channel/reference/sampling/participant-session evidence is recorded. Sample success cannot replace complete integrity. |
| `present_unverified` | Positive local path/content evidence exists, but release identity, license/access, completeness, integrity, or parsability remains unresolved and no stronger `restricted`, `corrupt`, or `partial` rule is established. |
| `missing` | Phase-I is COMPLETE; the targeted full-root locator covers all admitted mounts; an authoritative candidate marker/member set exists; no local release, cache, archive, partial transaction, mirror, or symlink-target copy matches; and the final destination is absent. Phase-I no-hit alone is insufficient. |
| `unknown` | Evidence is insufficient, contradictory, stale, unsupported, or globally PARTIAL with no positive candidate hit. `unknown` never authorizes download. |

If both corruption and missing members are observed in an anchored release, record both facts and
use `corrupt` as the primary status because known present bytes fail integrity; do not hide the
missing-member evidence. If legal authority is absent, `restricted` takes precedence and content
inspection stops.

## Registry generation and semantic validation

After all four targeted jobs reach immutable terminal states, submit one `cpu` registry-validation
job in `eeg2025`. The controller must first observe and hash-bind every targeted terminal marker,
`status.json`, and artifact manifest. Legitimate nonzero `BLOCKED`, `PARTIAL`, `FAILED`,
`STALE_INPUT`, `restricted`, or `corrupt` audit outcomes remain admissible terminal evidence for
drafting the corresponding bounded registry state; an `afterok`-only chain must not suppress the
validator. Use audited `afterany` semantics if the site supports them, or submit only after the
controller has observed all terminal states. Scheduler dependency alone is never admission:
the validator must match the exact terminal status and artifact hashes in its frozen request.
A missing, unsafe, or unverifiable terminal marker blocks registry publication. The validator
creates a draft for every candidate ID, including `unknown` or `restricted`; absence of success
is not grounds to omit a record.

The generator maps only evidence pinned in targeted `artifacts_manifest.json`:

- `official_version` comes from authenticated release/entity metadata, otherwise `null`.
- `discovered_paths` contains only curated, normalized, in-root release paths; candidate substring
  matches and participant-bearing member paths are excluded.
- `file_count` and `byte_count` come from the anchored full integrity manifest, otherwise `null`.
- `hash_strategy` names the official algorithms, local SHA-256 coverage, skipped bytes, manifest
  hashes, and why official equivalence is or is not established.
- `sample_read_result` distinguishes passed, failed, unsupported, not attempted, and bounded
  coverage; it never says the release is fully readable from a sample alone.
- field, channel/reference/sampling, and participant/session objects contain evidence status,
  artifact hashes, and explicit unknowns rather than guessed values.
- `inventory_job_id` is `918918`; `reviewed_at_utc` is registry review time, not source publication
  or file modification time.

Validate the existing JSON Schema and additional semantic rules: exact ID set; no extra record;
absolute paths below the fixed root with safe ancestry; unique paths; source anchor allowlist;
version/source consistency; no raw participant ID; counts equal the full manifest; hashes resolve;
and the seven-state prerequisites above. In particular, reject `verified_available` when any
version, access, full-integrity, required parser-stratum, or field-provenance prerequisite is
missing, and reject `missing` without COMPLETE Phase-I plus exhaustive targeted no-hit evidence.
All four records in one publication set must bind exactly one admitted Phase-I
job/attempt/state/manifest snapshot. Reject evidence that mixes a PARTIAL snapshot with a later
COMPLETE attempt, combines shards from different Phase-I attempts, or splices incompatible source
revisions. Within that set, each dataset record binds one internally consistent source revision;
historical provenance may be cited but cannot supply members or checksums to the selected revision.

Publish `datasets/registry/<dataset_id>.json` by no-replace only after semantic validation. If a
final registry file appears concurrently, stop, hash and review it; never merge or overwrite it.
Save the registry-validation status and draft/final hashes under a job-scoped report directory.

## Safe-download authorization

A new download transaction is legal only when all of the following hold:

1. The registry status is evidence-backed `missing`, not `unknown`, `present_unverified`,
   `partial`, `restricted`, or `corrupt`.
2. Phase-I was COMPLETE and the targeted locator ruled out existing final, cache, archive,
   partial, mirror, and declared in-root symlink-target copies.
3. Exact source, release/entity version, data license/access path, official expected member list,
   checksums when available, and credentials channel are frozen and approved.
4. A Slurm preflight verifies destination ancestry, free space for download plus safe extraction,
   inode budget, expected network bytes, checksum time, archive expansion bound, and that the final
   directory does not exist.
5. Download, validation, safe member-list inspection, extraction, and atomic publication use
   separate recorded Slurm jobs under the fixed data root's versioned `.partial` transaction.

Every preflight, network transfer, checksum pass, archive-member inspection, extraction, and
publication payload must be submitted through `scripts/slurm/submit.sh`, run in the registered
`eeg2025` environment, and use only the frozen `cpu` or `cpu-high` profile appropriate to its
audited estimate. There is no login-node execution and no environment/profile fallback. Final
publication stays on the same filesystem and uses an atomic no-replace primitive; it must never
overwrite an existing final path, force extraction, run `rsync --delete`, or modify raw files in
place. If the final path appears concurrently, stop publication and perform a new read-only audit.

An existing approved `.partial` transaction may be resumed only from its original frozen source
manifest and provenance; it is not reclassified as `missing` to start a different download. Any
source drift, credential failure, checksum mismatch, archive escape/symlink member, expansion
overrun, disk shortfall, or concurrently appearing final directory stops publication. No Phase-II
sample success, source-code checkout, synthetic fixture, or alternate mirror can authorize a
download.
