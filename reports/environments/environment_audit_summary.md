# Registered environment audit

Both strict audits completed through Slurm. No environment was created, upgraded, or modified. The
first-generation jobs `918736`/`918737` are preserved as provenance-incomplete observations. Jobs
`918770`/`918771` verified the environments against an earlier bundle and are retained as stale
prior-bundle evidence. Jobs `918774`/`918775` were then registered after control-plane validation
`918773` passed on HEAD `6442781fcaa06dec22f79abbd1be72f840e0f652`. Attachment job `918777`
exposed an NFS publication incompatibility, requiring another contract/job-bundle correction, so
that pair is also retained only as compatible prior-bundle evidence. After the corrected bundle was
frozen at `a5562aaf25ab7a0e0b7b4009afb421ecc2d4f049`, control validation `918787` passed and the
unchanged strict pair `918788`/`918789` completed. Parent attachment job `918791` then exposed a
READY-field spelling mismatch in the contract bundle. The pair remains compatible evidence for
that prior bundle but is no longer downstream authority. After the one-key correction was frozen at
`b5147c63bd1c43d87644ba69b315f6440e6d46f8`, control validation `918792` passed and the unchanged
strict pair `918793`/`918794` completed; these are the registered downstream audits.
Subsequent deterministic SIGSEGV failures in PDF child jobs required adding fail-closed diagnostic
instrumentation to the child Slurm payload. That job-bundle change makes `918793`/`918794`
compatible prior-bundle observations rather than current downstream authority. After control
validation `918804`, the unchanged pair `918805`/`918806` completed; these are the registered
downstream audits for the instrumented bundle.
Instrumented child `918809` then localized its SIGSEGV to native PyMuPDF module creation. Moving
that import out of non-rendering helper actions and adding `fitz` to the strict critical-import
probe changes the contract bundle; `918805`/`918806` are now prior-bundle evidence and a new audit
pair was required. Control validation `918814` passed, followed by strict audits
`918815`/`918816`; these were registered for the lazy-import bundle. Post-registration validation
`918817` and parent `918818` completed, but lazy CPU child `918819` again crashed in native PyMuPDF
module creation. Routing only the registered PDF renderer to the L40S profile changes the audited
contract/job/environment bundle. The pair is therefore prior-bundle evidence and an exact strict
rerun is pending; neither environment was modified.

Control validation `918821` then passed on commit `76ef67f`, followed by strict audits `918822`
and `918823`. Both completed with provenance and unchanged locks; these are now the registered
authorities for the L40S-renderer bundle. Post-registration validation `918824` passed, but a
subsequent fail-closed dependency/post-submit-resource validation correction changes the contract
bundle before any parent attachment job ran. The pair is now compatible prior-bundle evidence and
an unchanged strict rerun is pending.

Control validation `918825` then passed on commit `190285d`, followed by strict audits `918826`
and `918827`. Both completed with provenance, unchanged locks, and no compatibility failure; these
are now the registered authorities for the hardened L40S-renderer bundle.
Post-registration validation `918828` and parent `918829` completed, but L40S children
`918830`/`918831` deterministically failed closed with `OSError` during parent revalidation. Adding
bounded diagnostic evidence changes the contract bundle; `918826`/`918827` are now compatible
prior-bundle observations and an unchanged strict rerun is pending. Neither environment changed.
Diagnostic-bundle-v1 audits `918833` (CPU) and `918834` (L40S) subsequently completed with the same
explicit/pip lock hashes and required allocations. A pre-use security review tightened the
diagnostic schema, making those two jobs prior-bundle observations as well; no environment was
modified and a final exact rerun remains pending.
Control validation `918835` passed on the tightened bundle. Final strict audits `918836` (CPU) and
`918837` (L40S, `afterok:918836`) then completed with provenance, unchanged locks, exact registered
resources, and no compatibility failure. They are the current registered environment authorities;
a separate post-registration validation is still required before attachment submission.

| Environment | Strict Slurm job / actual allocation | Python | Verified capability | Explicit / pip lock SHA-256 | Status |
|---|---|---|---|---|---|
| `eeg2025` | `918836`; `CPU`; `nodecpu05`; 2 CPU; 8 GiB; 30 s | 3.13.7 | NumPy 2.4.4, SciPy 1.17.0, MNE 1.11.0, h5py 3.15.1, pandas 3.0.1, sklearn 1.8.0; all critical imports passed | `cc644eea…9d` / `ad6370f7…c0207` | compatible; registered CPU/data-audit role verified |
| `icml` | `918837`; `afterok:918836`; `L40S`; `node39`; 8 CPU; 64 GiB; 1 GPU; 12 s | 3.9.25 | PyTorch 2.8.0+cu128; PyMuPDF/fitz 1.26.5; CUDA available; cuDNN 91002; one NVIDIA L40S; scheduled tensor operation passed | `2c04fc17…f8a1` / `7af84a80…9a939` | compatible; registered GPU/model and PDF-renderer role verified |

The strict capture found 149 explicit Conda entries and 247 pip entries for `eeg2025`, and 34/113
respectively for `icml`. Sanitized lock files are replayable; raw stdout/stderr were hashed in
memory, stderr text was suppressed, and the high-confidence sanitizer recorded no non-URL secret
patterns. Current L40S audit `918837` recorded one visible NVIDIA L40S and a successful CUDA
tensor operation. The CPU audit correctly saw no CUDA device and does not claim GPU compatibility.

Both status files report `completed`, `provenance_complete=true`, and exit `0`; controller state was
also `COMPLETED/0:0`. Exact cluster, environment-at-submission, job, submitter, contract bundle,
Slurm job bundle, request, payload, HEAD, branch, remote, and within-job start/end state hashes were
retained. The dependency was preserved in the pre-sbatch/post-submit records even though the live
controller displayed `Dependency=(null)` after it had been satisfied.

The dirty-worktree binary-diff hashes differ between the CPU and GPU nodes even though HEAD and all
consumed config/job/bundle hashes match. The cause was not proven (node-specific Git/binary-patch
encoding is possible), so those dirty-diff hashes are not asserted equal. No scientific artifacts
are combined across them; each environment conclusion is bounded to its own recorded snapshot.

## Initial observations retained for audit history

| Environment | Slurm job / allocation | Python | Key verified capability | Explicit lock SHA-256 | Status |
|---|---|---|---|---|---|
| `eeg2025` | `918736`; `CPU`; node `nodecpu10`; 2 CPU; 8 GiB | 3.13.7 | Initial observation: NumPy 2.4.4, SciPy 1.17.0, MNE 1.11.0, h5py 3.15.1 | `cc644eead3ffa906a70573727b82974da57145b408b67a920741a4288fc0296d` | provenance incomplete; strict re-audit pending |
| `icml` | `918737`; `L40S`; node `node39`; 8 CPU; 64 GiB; 1 GPU | 3.9.25 | Initial observation: PyTorch 2.8.0+cu128, CUDA available, cuDNN 91002, one NVIDIA L40S visible | `2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1` | provenance incomplete; strict re-audit pending |

The initial CPU audit correctly saw no CUDA device; its installed PyTorch build is not used as evidence of GPU compatibility. Full initial artifacts remain under:

- `reports/environments/eeg2025/jobs/918736/`
- `reports/environments/icml/jobs/918737/`

Current strict artifacts are under `reports/environments/eeg2025/jobs/918836/` and
`reports/environments/icml/jobs/918837/`; prior-bundle artifacts remain under the corresponding
`918833`/`918834`, `918826`/`918827`, `918822`/`918823`, `918815`/`918816`, `918805`/`918806`,
`918793`/`918794`, `918788`/`918789`,
`918774`/`918775`, and `918770`/`918771` paths. Because `sacct` is unavailable, completion is evidenced by
each payload's no-replace `status.json`, live allocation capture, and controller state observed with
`scontrol`; unavailable historical accounting fields are not inferred.
