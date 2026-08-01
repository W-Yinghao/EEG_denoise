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
