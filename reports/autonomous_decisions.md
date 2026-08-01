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

## 2026-08-01: explicit cross-environment PDF handoff

- Evidence: the registered `eeg2025` audit confirms the CPU/data environment but the earlier
  attachment attempt showed it lacks the complete PDF extraction/rendering stack. The registered
  `icml` environment contains PyMuPDF. The hardened parent contract originally required every
  attachment extraction to succeed while the child contract required the PDF to appear in that
  parent's manifest, making the legal two-environment path impossible.
- Decision: add an explicit `--defer-pdf-to-registered-renderer` mode. The parent CPU job still
  opens, snapshots, hashes, and manifests the PDF, but records exactly one deferred renderer
  (`extract_pdf` in `icml`). The dependent child accepts only that parent status, source hash, and
  manifest binding. It does not accept an omitted or generically unsupported PDF.
- Alternatives rejected: installing PDF packages, dropping the PDF from the manifest, weakening
  the parent-success check, or letting an expected failure stand. Those options would change a
  shared environment or break attachment completeness/provenance.
- Impact: the contract/job bundle changed after audits `918770`/`918771`, so both were immediately
  marked stale for downstream use. After commit `6442781` and successful Slurm control validation
  `918773`, the unchanged strict rerun completed as `918774` (`eeg2025`) followed by dependent
  `918775` (`icml`); those were registered until the later NFS publication correction changed the
  bundle again.

## 2026-08-01: preserve failed attachment job and replace NFS directory publication

- Evidence: parent attachment job `918777` exited `3` and is preserved. Its text attachment left a
  complete hash-bound `READY` tree at the final `renameat2(RENAME_NOREPLACE)` call, while the target
  directory was absent; the exact errno was not persisted and is not inferred. The code root is
  NFSv3. The same job also correctly reported that the source ZIP contained currently unsupported
  members: five PDF, five SVG, and three PowerShell files. The top-level PDF was safely deferred.
- Decision: do not fall back to ordinary rename, which could overwrite across a race on NFS. Each
  job now exclusively creates its job-scoped final directory and writes every artifact with
  `O_EXCL`; consumers require the terminal, hash-bound `COMPLETE` marker plus successful job status.
  A crash may leave an incomplete directory, but no consumer accepts it and job IDs are never
  reused. ZIP PDFs are opaque, hash/header/MIME-verified deferred members for registered `icml`
  render jobs; SVG and PowerShell are bounded text and are never rendered or executed by the CPU
  parent.
- Alternatives rejected: weakening no-overwrite semantics, ignoring ZIP members, treating PDF as
  an ordinary image, or deleting/reusing `918777` artifacts.
- Impact: audits `918774`/`918775` became bundle-stale and were immediately deregistered. They must
  be rerun after the corrected bundle passes scheduled syntax/control validation.

## 2026-08-01: register the unchanged post-correction environment audit pair

- Evidence: after commit `a5562aa`, CPU control validation `918787` completed with machine status
  `passed` and zero failures. Strict audit `918788` then completed for `eeg2025` on `CPU`, followed
  by dependent audit `918789` for `icml` on `L40S`; both status files report `completed`,
  `provenance_complete=true`, and exit `0`. Their explicit and pip lock hashes equal the previously
  observed values, and neither environment was modified.
- Decision: register `918788` and `918789` as the current downstream audit authorities, while
  retaining every earlier audit as prior-bundle or provenance-incomplete evidence.
- Impact: attachment and inventory jobs may consume these exact audit IDs. Any later change to the
  audited contract/job/submitter/cluster bundle invalidates this authority and requires another
  strict pair.

## 2026-08-01: preserve failed parent verification and correct its READY binding key

- Evidence: parent attachment job `918791` safely extracted the three source attachments and wrote
  a 52-record, 4,167,384-byte closed artifact manifest, but exited `3` during its own verifier and
  never wrote `EXTRACTION_COMPLETE.json`. Its persisted READY marker used
  `artifacts_manifest_sha256`, while the generating helper's verifier requires the canonical
  `artifact_manifest_sha256`; the stored totals themselves agree with the three manifest trees.
- Decision: preserve all `918791` artifacts as failed, change only the generating key to the
  verifier's already established singular spelling, and do not let any child renderer consume the
  incomplete parent.
- Impact: this one-line contract-bundle change makes `918788`/`918789` stale. They are deregistered
  until another unchanged CPU→L40S strict audit pair succeeds after Slurm control validation.

## 2026-08-01: register strict audits after the READY-key correction

- Evidence: control validation `918792` passed with zero failures on commit `b5147c6`. Strict
  `eeg2025` audit `918793` completed on `CPU`, followed by dependent `icml` audit `918794` on
  `L40S`; both are provenance-complete, exit `0`, retain the registered lock hashes, and report no
  compatibility failures.
- Decision: register `918793`/`918794` as the current downstream audit pair without changing either
  environment.
- Impact: a new parent attachment job may bind these exact IDs after a post-registration control
  validation. Earlier jobs remain immutable prior-bundle or failed evidence.

## 2026-08-01: instrument deterministic PDF renderer SIGSEGV before changing behavior

- Evidence: child jobs `918797`–`918802` all exited `139` at the `icml` PDF helper invocation on
  CPU nodes 10/11. A single unchanged retry, `918803`, reproduced exit `139` on nodecpu10. None
  produced a job-private PDF snapshot, output tree, or COMPLETE marker; stderr identifies only the
  shell line and SIGSEGV. Earlier job `918742` shows that PyMuPDF once ran in this environment on a
  CPU allocation, but its older contract is not evidence for the current path.
- Decision: retain every failed job and enable Python's built-in faulthandler in the registered
  `extract_pdf` Slurm payload. Do not alter inputs, parser rules, budgets, renderer, or parent
  bindings until a scheduled reproduction records a stack.
- Impact: the Slurm job bundle changes, so audits `918793`/`918794` and parent `918796` become stale
  for new children. A fresh control/audit/parent chain is required before the instrumented retry.

## 2026-08-01: register the instrumented diagnostic audit pair

- Evidence: validation `918804` passed with zero failures on commit `60acf0d`. Strict audits
  `918805` (`eeg2025`/CPU) and dependent `918806` (`icml`/L40S) both completed with provenance,
  exit `0`, unchanged lock hashes, and no compatibility failures.
- Decision: register `918805`/`918806` for the faulthandler-instrumented bundle; the diagnostic does
  not alter either Conda environment.
- Impact: after a post-registration control validation, a fresh parent and one instrumented child
  may run. The diagnostic child cannot become a successful renderer unless its own COMPLETE closes.

## 2026-08-01: isolate PyMuPDF from non-rendering helper actions

- Evidence: instrumented child `918809` reproduced exit `139`. Its fatal Python stack ends at
  `extract_pdf_pymupdf.py:18` while importing `fitz`, inside `pymupdf/mupdf.py` native module
  creation. No parent validation, PDF snapshot, content parsing, or rendering frame appears. The
  same job's prior `contract` helper process completed even though the old module imported `fitz`
  at top level, proving that import success is not sufficient to authorize an unnecessary second
  native import process.
- Decision: remove the top-level native import. Pure contract/verify/finalize actions remain
  renderer-free; only `extract` lazily imports PyMuPDF after the pure-Python parent binding,
  snapshot/header checks, output allocation, and disk check. Add `fitz` to the strict `icml`
  critical-import audit with faulthandler enabled.
- Impact: no input, output, PDF safety budget, parent binding, or environment package changes. The
  contract bundle changes, so audits `918805`/`918806` and parent `918808` become stale; a fresh
  chain is mandatory before testing whether lazy import resolves the crash.

## 2026-08-01: register strict audits for lazy PyMuPDF loading

- Evidence: control validation `918814` passed on commit `070c3e5`. Strict audits `918815`
  (`eeg2025`/CPU) and `918816` (`icml`/L40S, `afterok:918815`) completed with provenance, exit `0`,
  unchanged lock hashes, and no compatibility failures. The `icml` probe explicitly imported
  fitz/PyMuPDF 1.26.5 and completed its CUDA tensor check.
- Decision: register `918815`/`918816` for the lazy-import bundle without modifying either
  environment.
- Impact: a post-registration control validation, fresh parent, and one CPU child are still needed;
  the L40S import audit alone is not evidence that the CPU renderer crash is fixed.

## 2026-08-01: route the registered PDF renderer to L40S after CPU-native failure

- Evidence: post-registration validation `918817` and parent attachment job `918818` completed
  successfully. Lazy-renderer child `918819` then exited `139` on `CPU` node `nodecpu11` after its
  pure-Python contract, parent binding, source snapshot, PDF header, output allocation, and disk
  checks had completed. Faulthandler again places the fatal frame in native module creation under
  `pymupdf/mupdf.py`, reached from `load_fitz_renderer`; it produced no renderer output or COMPLETE
  marker. Together with failures `918797`–`918803` and `918809`, this reproduces on CPU nodes 10/11.
  By contrast, the same frozen `icml` lock imported fitz/PyMuPDF 1.26.5 successfully in strict L40S
  audit `918816` and completed that job's CUDA compatibility check.
- Decision: retain every failed CPU child and bind `extract_pdf` to the already audited `L40S`
  profile, including the parent defer record, pre-sbatch request, live allocation, GRES, parser,
  status, and child contract. The CPU parent remains on `eeg2025`; only the registered `icml` PDF
  renderer moves to L40S. No package, PDF budget, scientific configuration, or source attachment is
  changed.
- Alternatives rejected: further identical CPU retries have already exceeded the single unchanged
  retry allowance; changing or installing renderer packages is unauthorized; weakening the native
  crash into an apparent success would invalidate the attachment evidence.
- Impact: this is an infrastructure compatibility recovery, not a scientific GPU result or a
  latency comparison. The contract/job/environment registry changed, so audits `918815`/`918816`,
  validation `918817`, and parent `918818` are prior-bundle evidence. A new control validation,
  strict CPU→L40S audit pair, post-registration validation, parent, and L40S child are required.

## 2026-08-01: register strict audits for the L40S renderer route

- Evidence: control validation `918821` passed with zero failures on commit `76ef67f`. Strict
  `eeg2025` audit `918822` completed on `CPU` node `nodecpu10`, followed by dependent `icml` audit
  `918823` on L40S node `node39`; both are provenance-complete with exit `0`. Their explicit and pip
  hashes remain `cc644eea…9d`/`ad6370f7…c0207` and `2c04fc17…f8a1`/`7af84a80…9a939` respectively.
  The L40S probe again imported fitz/PyMuPDF 1.26.5 and completed its CUDA tensor check.
- Decision: register `918822`/`918823` as the current environment authorities for the exact L40S
  renderer bundle. No Conda package or environment path changed.
- Impact: a separate post-registration control validation is still mandatory before the parent
  attachment job; successful environment import is not yet a successful PDF extraction.

## 2026-08-01: close attachment dependency and post-submit resource ambiguity

- Evidence: post-registration control validation `918824` passed on commit `e3ed1a6`. Before any
  parent attachment submission, an independent static review found that the downstream attachment
  contract accepted a correct dependency from either the Slurm environment or live allocation even
  if the other source carried a non-null conflicting value. It also hash-bound the post-submit
  manifest but did not compare that manifest's duplicated resource fields one by one with the
  validated pre-sbatch request. No observed job exploited either gap.
- Decision: reject every non-null dependency value other than the exact `afterok` job and only then
  classify dependency visibility; require the post-submit partition/account/QOS/CPU/memory/time/
  GRES/constraint/signal fields to match the frozen job contract; require integer CPU fields and
  one node, one task, and `KillOInInvalidDependent=Yes` in the live allocation.
- Impact: these are provenance hardening changes, not environment or scientific changes. They alter
  the contract bundle, so `918822`/`918823` and validation `918824` are prior-bundle evidence. A new
  control validation and unchanged strict CPU→L40S audit pair are required before attachment work.

## 2026-08-01: register audits after attachment provenance hardening

- Evidence: control validation `918825` passed with zero failures on commit `190285d`. Strict audit
  `918826` completed on `eeg2025`/CPU and dependent audit `918827` completed on `icml`/L40S, both
  provenance-complete with exit `0`, unchanged explicit/pip hashes, and no compatibility failures.
  The L40S probe again imported fitz/PyMuPDF 1.26.5 and completed its CUDA tensor check.
- Decision: register `918826`/`918827` as current environment authorities without modifying either
  environment. Require a fresh post-registration control validation before attachment submission.
- Impact: the exact hardened bundle can proceed only after that final administrative check.

## 2026-08-01: diagnose deterministic cross-node parent revalidation failure

- Evidence: post-registration validation `918828` passed and parent attachment job `918829`
  completed with six L40S-bound deferred PDFs, zero credential findings, and a closed parent
  manifest. Top-level L40S child `918830` passed its request/environment/allocation contract but
  failed with controlled exit `3`/`OSError` before creating its own source snapshot. One unchanged
  retry, `918831`, reproduced the same phase and exit on node `node39`; no renderer output or
  COMPLETE marker exists for either job. The parent ran on `nodecpu10` and recorded the live source
  `st_dev=53`. Static review identifies `verify_original_source`'s full identity equality—including
  client-local `st_dev`—as the highest-confidence cross-node failure source, but this remains an
  inference until the failing field is captured on node39.
- Decision: stop unchanged retries. Add a bounded diagnostic record containing only the controlled
  error type/detail, repository-relative traceback locations, and identity mismatch field names;
  suppress any detail matching credential patterns. Do not render, relax identity checks, or submit
  the five embedded PDFs until one audited diagnostic child identifies the failing condition.
- Impact: the helper/contract bundle changes, so audits `918826`/`918827`, validation `918828`, and
  parent `918829` are prior-bundle evidence. A fresh control/audit/register/validation/parent chain
  is required before the one diagnostic child.

The first diagnostic-bundle validation/audit chain (`918832`, `918833` → `918834`) completed with
zero control failures, unchanged environment locks, and the expected CPU/L40S allocations. A
pre-child static safety review then found that persisting generic exception text could disclose an
unrecognized secret and that diagnostic construction could mask the original failure. The
diagnostic was therefore revised before use: it records only a fixed code, built-in/custom error
type, numeric errno, allowlisted identity fields, and repository-relative traceback locations; its
entire best-effort writer cannot replace the original exception. Jobs `918832`–`918834` are retained
as prior-bundle evidence, and no attachment child used that first diagnostic format.

## 2026-08-01: register tightened-diagnostic environment authorities

- Evidence: validation `918835` passed on commit `06cf50e`. Audit `918836` completed on
  `eeg2025`/CPU (`nodecpu05`, 2 CPU, 8 GiB), and its dependent `918837` completed on `icml`/L40S
  (`node39`, 8 CPU, 64 GiB, one L40S). Both report `provenance_complete=true`, exit `0`, unchanged
  explicit/pip locks, and exact request/allocation fields; the dependency remains in job environment
  and submission evidence after Slurm cleared the live satisfied dependency.
- Decision: register `918836`/`918837` as the current authorities without changing either
  environment. Require one post-registration control validation before the parent attachment job.
- Impact: renderer diagnostics may proceed only from a fresh parent bound to these IDs after that
  validation; no prior parent or child result is promoted into the new chain.

## 2026-08-01: use closed parent artifacts across scheduler nodes

- Evidence: post-registration validation `918838` passed. Fresh parent `918839` completed on CPU
  with six L40S deferred PDFs, a closed artifact tree, and zero credential findings. Its sole
  diagnostic child `918840` ran on L40S/node39 and failed closed before its own snapshot with the
  fixed diagnostic code `parent_original_source_identity_mismatch`; the only mismatched allowlisted
  field was `device`. The traceback ends at `verify_original_source`, confirming that CPU-recorded
  `st_dev` is not portable to the GPU node while all parent artifact hashes remain closed.
- Decision: preserve full live-source identity checks within the parent job. For a dependent PDF
  renderer, validate the complete parent artifact/hash closure without reopening live originals,
  select the registered top-level parent snapshot or registered extracted ZIP-member artifact, and
  create a new child-private snapshot from that closed source before PyMuPDF is loaded. The submitted
  original/member path remains a manifest-bound selector, not renderer input.
- Impact: `918835`–`918840` are retained as direct root-cause evidence but are prior-bundle inputs.
  No PDF was rendered by `918840`, and no embedded child was submitted. The fix requires another
  exact control/audit/register/validation/fresh-parent chain before retrying one top-level child.

## 2026-08-01: register closed-parent-artifact runtime authorities

- Evidence: control validation `918842` passed on commit `e819347`. Strict audit `918843`
  completed on `eeg2025`/CPU and dependent audit `918844` completed on `icml`/L40S. Both report
  complete provenance, exit `0`, exact registered requests/allocations, and unchanged explicit/pip
  locks; `918844` imported PyMuPDF and completed its CUDA tensor probe on one L40S.
- Decision: register `918843`/`918844` as the current environment authorities without modifying
  either environment. Require a post-registration validation before generating a fresh parent.
- Impact: only a parent produced after that validation may authorize the top-level renderer retry.

## 2026-08-01: require cold-start renderer compatibility

- Evidence: post-registration validation `918845` passed and fresh parent `918846` completed with
  `live_sources_verified=true`. Top-level children `918847` and the single unchanged retry `918848`
  both passed parent validation, created a child-private snapshot with the registered source hash,
  then exited `139` at `pymupdf.mupdf` native module creation before opening the PDF. Both ran on
  L40S/node39 with the registered `icml` locks. By contrast audit `918844` imported fitz only inside
  `conda run` after preloading other native packages, so it did not prove renderer-equivalent cold
  start compatibility.
- Decision: stop unchanged retries. Disable core dumps for renderer/audit jobs; add two independent
  cold-start import probes to the `icml` audit (direct interpreter and registered `conda run`) while
  retaining only fixed exit codes and bounded metadata. Require the `conda run` cold start to pass,
  and launch the formal native renderer through that registered mode, as prescribed by the
  environment contract. No package or environment mutation is allowed.
- Impact: `918842`–`918848` are prior-bundle evidence; neither failed child produced a PDF render.
  The cold-start change requires a new validation/audit/register/validation/fresh-parent chain.

Validation `918849` passed and audit `918850` completed, but dependent L40S audit `918851` failed
as designed: in fresh processes, both direct and `conda run` no-preload imports exited `139`, while
the separate full runtime probe still imported fitz successfully. This disproves Conda activation
alone as the remedy, so `918850` is prior-bundle evidence and `918851` is retained as a failed
diagnostic. Before changing the formal renderer again, the audit is extended with a frozen matrix
under the same direct registered interpreter: CPU-stack prefixes, CPU/Torch interactions, Torch
with and without the runtime CUDA warmup, and exact full runtime order. Every plan runs twice in a
new serial process with core dumps disabled; only fixed exit codes and successful bounded metadata
are retained, and preloads must not import fitz transitively. The diagnostic requires the exact
full-CUDA positive control to reproduce twice, but it is not an authority for a formal renderer
until a stable minimal plan is selected and re-audited.

Validation `918853` passed on that matrix bundle and CPU audit `918854` completed. Dependent L40S
audit `918855` failed closed after recording all 32 direct-interpreter cells: every preload plan,
including both exact-order `full_cuda` replicates, exited `139`; the ordinary `conda run` runtime
probe in the same allocation still passed. This proves that direct-interpreter preload alone is
insufficient under the tested strict-warnings condition, but it does not isolate activation from
warnings or stream handling. The next audit therefore freezes one probe implementation in a
launcher `{direct, conda_run}` × preload `{none, runtime_full_cuda}` × warnings
`{default, error}` design,
with two new processes per cell. Each process retains separately bounded stdout/stderr and writes
an atomic pre-import marker after its preload and immediately before `fitz`; no formal renderer is
authorized unless the preregistered `conda_run/runtime_full_cuda/default` positive control succeeds
twice. The full preload reuses the ordinary runtime probe's module order, version reads, and exact
Torch/CUDA inspection; direct cells have Conda activation variables removed, while `conda_run`
cells must establish the registered prefix themselves. A separate pure-Python verifier requires
the exact 16-cell filename/hash set and parses both positive-control markers/results rather than
trusting exit codes or file existence alone.
If it succeeds, a separate prospective matrix will identify the minimal sufficient preload before
formal behavior changes. No environment or attachment input is modified.
The formal `extract_pdf` payload also has an explicit `diagnostic_pending` startup guard, so an
accidental submission cannot reach parent validation, source snapshots, or PyMuPDF while this
screen and the subsequent minimal-plan freeze remain incomplete; it exits through the job status
trap as `blocked_startup_authorization`.

## 2026-08-01: freeze one shared renderer startup contract

- Evidence: control validation `918884` passed on commit `a96b6aa`. The first CPU-audit submission
  `918885` was mistakenly given `afterok:918884`; its own request validator rejected that dependency
  immediately, before any environment probe, and dependent `918886` was cancelled as
  `DependencyNeverSatisfied`. The corrected no-dependency CPU audit `918887` completed on
  `nodecpu04` with unchanged locks and full provenance. Its L40S child `918888` retained the exact
  16-cell diagnostic but failed the preregistered positive control. Both direct/default/no-preload
  replicates and both direct/default/full-preload replicates imported PyMuPDF `1.26.5`; all four
  direct/`warnings=error` cells reached the pre-import marker and then exited `139`, regardless of
  preload. All eight `conda_run` cells failed before the marker because the diagnostic launcher
  cleared `CONDA_PREFIX`/`CONDA_DEFAULT_ENV` while retaining `CONDA_SHLVL=1`, causing Conda 22.9.0
  activation to fail. Those Conda cells are invalid launcher evidence, not renderer failures.
- Decision: do not promote any successful exploratory cell. Freeze the single prospective contract
  `conda_standard_default_none`: standard registered `conda run`, default Python warnings, no
  deliberate preload, PyMuPDF `1.26.5`, two independent processes. Put its canonical contract hash
  in the tracked environment registry before audit. Probe and formal extraction now share one
  fail-closed loader; the formal path no longer injects `PYTHONWARNINGS=error` and no longer uses a
  source-edited authorization switch. Runtime evidence must be revalidated and its validation hash
  pinned in `configs/environments.yaml` before any parent or PDF child can run.
- Impact: `918887`/`918888` are prior-bundle diagnostic evidence only; neither is registered as an
  environment or renderer authority. No PDF render, data scan, model, gate, or scientific result was
  produced. A fresh control validation and exact CPU → L40S audit pair are required for the shared
  startup bundle; registration may change only environment audit IDs/status/lock fields and the
  renderer validation hash, followed by post-registration validation.

## 2026-08-01: bound renderer streams and pin terminal attachment evidence

- Evidence: independent static review found that a formal renderer could declare empty streams
  without observing them, that child contracts did not explicitly compare their parent dependency,
  and that terminal validation/output checksum files were not rechecked by the success EXIT trap.
- Decision: bind both child extract/verify paths to `dependency_job_id` and
  `request_dependency=afterok:<parent>`. Capture formal renderer stdout and stderr through separate
  bounded FIFOs, accept only two empty streams and zero child/reader exit codes, and retain empty
  files plus an exact serialized validation record. Non-empty or failed streams are removed to avoid
  publishing possible credentials; status retains the pre-cleanup byte counts, hashes, process/reader
  exit codes, and `streams_retained=false`. Parent and child now pin terminal validation and output
  checksum hashes in memory, re-run the read-only verifier with bytecode writes disabled, check the
  checksum list before and after verification, and repeat the checks in the success EXIT trap.
- Residual assumption: job output directories are newly created with mode `0700` and are not modified
  by another process with the same Unix identity. The terminal verifier hashes every manifest leaf,
  but the shell checksum list pins the manifest roots rather than every page/text/snapshot leaf, so a
  malicious same-UID rewrite in the final verifier-to-status micro-window is not cryptographically
  excluded without an external immutable store or signing authority. Concurrent ordinary work is
  isolated by job-scoped paths; any observed change fails closed. Shared Conda Python source is also
  assumed not to be replaced during import; native renderer objects are additionally bound by mapped
  device/inode and same-descriptor hashes. These assumptions do not authorize scientific execution.

## 2026-08-01: register the prospective shared renderer audit

- Evidence: pending-state control job `918899` passed with no failures. Strict audit `918900`
  completed for `eeg2025` with no dependency; `918901` completed for `icml` under
  `afterok:918900`. Both retained exact locks and complete provenance. The L40S job observed one
  NVIDIA L40S and passed two independent canonical renderer starts with empty streams, identical
  startup/component fingerprints, and validation hash `8d38fa41…a04eec`.
- Decision: update only mutable environment registration fields. Set both `audit_job_id` and
  `strict_reaudit_job_id` to `918900`/`918901`, mark both environments compatible and verified,
  and register the exact renderer validation hash. Do not modify either Conda environment.
- Impact: attachment submission remains disabled until a distinct verified-state control job
  revalidates this registry and both audit evidence chains. Scientific stages remain blocked by the
  recorded semantic conflicts and have not been submitted.

## 2026-08-01: diagnose the first shared-startup PDF canary without weakening refusal rules

- Evidence: verified-state control job `918902` passed, and fresh attachment parent `918903`
  completed with the exact registered audit pair `918900`/`918901`. Its one top-level L40S canary,
  `918904`, passed parent binding and both shared renderer starts, then exited `3` after committing
  page 1–8 text and images. Page 8 text and PNG are byte-identical to the older complete extraction;
  page 9 was not committed. The page-8 PNG and failure marker differ in mtime by about 2 ms. The
  bounded formal stdout was empty; the non-empty stderr was destroyed by policy and only its byte
  count/hash remain. The v1 failure marker records only `ExtractionError`, so it cannot uniquely
  distinguish the page-8 warning audit from another explicit failure before page-9 publication.
- Decision: retain `918904` as failed evidence and do not submit its five sibling PDFs or retry the
  unchanged helper. Add fixed, allowlisted stage/page/counter diagnostics and repository-relative
  traceback locations bound to the helper hash. Drain PyMuPDF warnings after each registered PDF
  operation. Warning diagnostics retain only capped byte/line counts, an allowlisted credential
  pattern identifier if detected, and `raw_retained=false`; they retain neither warning text nor a
  guessable warning digest. Credential matches retain only the registered pattern identifier. No
  warning, credential, PDF safety, startup, or stream refusal is relaxed. The child shell pins the
  exclusive failure record hash into its no-replace status and rechecks it in the EXIT path.
- Impact: the contract and Slurm-job bundles changed, so `918900`/`918901`, `918902`, and `918903`
  are prior-bundle evidence for new work. Both environment registrations return to
  `pending_strict_reaudit`. One fresh control/audit/register/control/parent/canary chain is required
  to identify the terminal stage. Population base, P0, gates, and backups remain frozen.

## 2026-08-01: register the bounded-failure diagnostic audit pair

- Evidence: pending-state control job `918907` passed with zero failures on commit `b3a15d1`.
  Strict audit `918908` completed for `eeg2025` on CPU with no dependency; dependent audit `918909`
  completed for `icml` on L40S under `afterok:918908`. Both have complete provenance, exit `0`,
  compatible runtime probes, unchanged explicit/pip lock hashes, and the same immutable
  environment-policy hash. `918909` observed one NVIDIA L40S, passed its CUDA operation, and passed
  both shared PyMuPDF 1.26.5 cold starts with empty streams and exact component fingerprints. The
  renderer validation hash is `69b0fc57…02e4`.
- Decision: register only the two new audit IDs/status fields and the exact renderer validation
  hash; do not change either Conda environment. Preserve the recurring CPU/L40S difference in the
  hash of unrelated pre-existing dirty worktree files as recorded evidence; the Git commit,
  request, environment policy, contract bundle, Slurm-job bundle, configs, locks, and renderer
  components are exact and stable within each job.
- Impact: a distinct verified-state control validation is still required before a fresh attachment
  parent or diagnostic canary may consume these authorities. No data inventory or science job has
  been submitted.

## 2026-08-01: stop the unchanged PDF renderer after a bounded page-operation diagnosis

- Evidence: verified-state control `918915` passed. Fresh parent `918916` completed against audit
  jobs `918908`/`918909` and closed the same three attachment hashes with manifest SHA-256
  `4e6dff30…e425b`. Its only top-level canary `918917` passed request, parent, snapshot, allocation,
  and shared-startup validation, then failed closed at `page_text_warning_audit` on page 8. The
  marker records seven fully validated text/render pages, one 40-byte warning line, no credential
  pattern, no raw warning, and no cleanup failure. The shell pinned that marker as
  `0e0eba84…7e145`; formal stdout was empty and non-empty stderr was discarded by policy.
- Decision: do not ignore or classify the warning as benign without evidence; do not retry the
  unchanged child and do not submit the five embedded sibling PDFs. Preserve the earlier complete
  54-page Slurm semantic/visual review as content evidence, but mark the hardened supplement
  incomplete. Continue only the independent full-root read-only data inventory.
- Impact: attachment content review is complete with conflicts, while strict renderer provenance is
  terminally failed for this source/configuration. This does not authorize population base, P0,
  G1–G5, or B1–B6. Inventory job `918918` is the next legal stage and cannot turn a renderer failure
  into a scientific pass.

## 2026-08-01: retain the full inventory request while it waits for cpu-high priority

- Evidence: inventory job `918918` was submitted through the frozen `cpu-high` profile with
  `afterok:918908`, 8 CPU, 64 GiB, a five-day wall-clock bound, and the mandatory read-only fixed
  data root. Slurm reports `PENDING (Priority)` and currently estimates a start at
  `2026-08-05T00:40:00`; the estimate is scheduler state, not a promised allocation or completion.
- Decision: do not shorten the full-root scan, change partitions/resources, run it on the login
  node, or substitute a top-level/tiny inventory merely to obtain an earlier result. Complete
  report-only work before the scheduled start, then freeze the worktree for the scanner's
  start/end stale-input comparison.
- Impact: queue time is not a scientific or engineering failure. Dataset presence remains
  `unknown`; no registry is promoted, no download begins, and Stage A remains blocked. If Slurm
  revises the start estimate, monitoring follows the live job rather than this snapshot.

## 2026-08-01: separate first-party source metadata from local dataset verification

- Evidence: first-party pages establish selected source facts: Mendeley v1 identifies the requested
  Klados--Bamidis version and CC BY 4.0; the Eye-BCI data descriptor states CC0 for the Synapse data;
  the EEGdenoiseNet GIN dataset metadata states CC0; and the SGEYESUB authors link their processed
  dataset to OSF 2QGRD while the unauthenticated OSF page/API returned 403 in this review. None of
  those network observations proves a matching local file, checksum, field, channel layout, record
  duration, participant structure, or sample readability.
- Decision: record source metadata and its explicit unknowns separately. Do not copy a code license
  onto data, do not infer that OSF is restricted from an automated 403, and do not turn a public
  source anchor into `verified_available` or `missing` local state.
- Impact: Phase-I inventory must complete before any targeted local audit. Only a subsequent
  evidence-bound version/license/sample-read audit may create the four dataset registry records or
  authorize a transactional download decision.

## 2026-08-01: cancel exhaustive inventory and reduce the private-project harness

- Evidence: job `918918` spent 14,810 seconds visiting 2,423,427 entries in the 30 TB private NFS
  workspace, hashed 10.28 GB of allowlisted metadata and wrote 2.23 GB of JSON without locating any
  of the four dataset names in the partial traversal. The user clarified that exhaustive absence
  proof and industrial content-addressed provenance are not required.
- Decision: cancel only job `918918`; retain its small terminal evidence but never treat its partial
  no-hit as `missing`. Do not rerun the exhaustive scanner. Replace it with a bounded name-only
  locator, candidate-specific checks and direct transactional download when a public dataset name
  is not found. Keep Git and small human-readable dataset records as provenance; use official
  checksums when supplied, but do not locally hash every file.
- Impact: worktree/bundle hash rollovers, CAS registry publication, exhaustive manifests and repeated
  environment authority audits are retired for the active workflow. Raw-data read-only handling,
  Slurm execution, legal access, support/query leakage checks and scientific gate ordering remain.

## 2026-08-01: fix OSF pagination and resume the bounded SGEYESUB download

- Evidence: metadata jobs `919154`/`919157` used OSF's default 10-item pages and reported 178 rows
  but a lower aggregate byte count than the one-page official enumeration. Download attempt
  `919162` then stopped on an already-created `study02_p15_prep.set`, proving that unstable default
  pagination had repeated at least one path while omitting another. The job did not publish a final
  directory; its completed files remain in `.osf-2qgrd.partial-919162`.
- Decision: request `page[size]=100` for each study (every study has fewer than 100 entries), reject
  duplicate relative paths, restore the complete 178-file total `1,611,314,510`, and resume only
  existing files whose sizes match the current official listing. Do not delete or redownload valid
  partial files.
- Impact: the failed attempts remain visible, no data or bundle hashing is added, and final
  publication still requires all 178 paths plus the targeted MNE sample read.
