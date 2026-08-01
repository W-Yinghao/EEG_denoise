# Slurm control-plane audit

Audit time: 2026-08-01 (Europe/Paris; before payload submission)

Only login-node control-plane commands were used: `command -v`, `sinfo`, `scontrol`, `squeue`, `sacct`, and Slurm version queries. No project Python, environment import, data scan, test, preprocessing, training, evaluation, or plotting payload was run on the login node.

## Frozen resource mapping

| Logical profile | Observed partition | Observed capacity/features | Audited maximum time | Submission mapping |
|---|---|---|---|---|
| `cpu` | `CPU` | 10 nodes, 748 CPUs, no GPU GRES | 4 days | no GRES/constraint |
| `cpu-high` | `cpu-high` | 2 nodes, 192 CPUs, no GPU GRES | 5 days | no GRES/constraint |
| `A100` | `A100` | 11 nodes, 38 GPUs; node feature includes `A100,40G` | 1 day | `--gres=gpu:1`; model fixed by partition |
| `H100` | `H100` | 2 nodes, 6 GPUs; node feature includes `Hopper,H100` | 1 day | `--gres=gpu:1`; model fixed by partition |
| `L40S` | `L40S` | 5 nodes, 24 GPUs; node feature includes `L40S` | 1 day | `--gres=gpu:1`; model fixed by partition |

Slurm version is `24.11.7`; controller reports cluster `gpucluster`. The user association reports default account `c2s`; a live user job showed account/QOS `c2s/normal`. Target partitions are `UP`, allow the relevant group/account, and deny `qos-mm`. The submission defaults are frozen in `configs/cluster/slurm.yaml`; each job must still state explicit CPU, memory, GPU, and wall-clock requests.

## Warnings and blocked evidence

`sacct` is currently unusable:

```text
sacct: error: _open_persist_conn: failed to open persistent connection to host:localhost:6819: Connection refused
sacct: error: Sending PersistInit msg: Connection refused
sacct: error: Problem talking to the database: Connection refused
exit_code=1
```

Consequently, completed-job accounting and actual historical allocations cannot currently be verified through SlurmDBD. Live jobs can still be inspected with `squeue`/`scontrol`; every submitted payload will capture `scontrol show job` and allocation environment evidence while it is live. This warning does not authorize inventing missing accounting fields.

The Slurm client also reported a configuration-hash mismatch:

```text
HASH_VAL = Different Ours=0x72e31e83 Slurmctld=0x263790d7
SLURM_CONF = /etc/slurm/slurm.conf
```

The controller was `UP` and partition queries succeeded, so scheduling may proceed using the observed controller mapping. The mismatch remains an infrastructure warning in every affected run.

## Conda path precheck

Both registered paths exist, are owned by `yinwang:comelec`, contain `bin/python` and `conda-meta`, and are not replacement environment paths. This precheck proved only filesystem presence; the later strict runtime evidence is reported separately in `reports/environments/environment_audit_summary.md`.

## Hardened administrative validation

Slurm job `918768` ran the hardened administrative validator on profile `cpu` (`CPU`, node
`nodecpu11`, 2 CPUs, 8 GiB, 30-minute limit). The live allocation, pre-sbatch request, post-submit
link, exact payload-argument hash, submitter/job/config/bundle hashes, shell syntax, Python syntax,
disabled gate/backup configs, and JSON schemas all matched. Its machine-readable result is
`status=passed`, `failure_count=0`, and the controller later reported `COMPLETED`, exit `0:0`, after
13 seconds.

The job artifacts became visible from the login node roughly one minute after the controller had
marked the job complete, while Slurm stdout/stderr were empty. This is recorded as NFS metadata
visibility latency, not as missing execution evidence: the no-replace status, request validation,
allocation capture, and validation JSON subsequently appeared and agree with the controller. Job
`918768` binds the dirty pre-commit snapshot that it recorded; it is not evidence for a later clean
checkout. After local commits `eb227a8` and `9bc5286`, job `918769` repeated the same validation on
HEAD `9bc5286742c23c19577f528d4ae029df9f765b52` and also reported `passed`, zero failures, and
controller `COMPLETED/0:0` in 12 seconds. The repository still contains preserved unrelated dirty
user work, so neither job is described as a clean-checkout validation.

After the cross-environment attachment handoff was frozen in commit `6442781`, job `918773`
repeated the administrative validation on that exact HEAD. It reported `status=passed`,
`failure_count=0`; the controller reported `COMPLETED/0:0` after 14 seconds on `nodecpu11`. This is
the validator that authorized the now-prior-bundle strict environment audits `918774`/`918775`.

After the NFS-safe attachment publication correction was frozen in commit `a5562aa`, job `918787`
repeated the validator on `CPU` node `nodecpu11` with 2 CPUs and 8 GiB. Its machine-readable result
is `status=passed`, `failure_count=0`; the controller reported `COMPLETED/0:0` after 13 seconds.
This validator authorizes the current strict environment chain `918788` → `918789`; registration
of those IDs is followed by a separate Slurm control validation rather than being inferred here.

Registration validation `918790` subsequently passed with zero failures on HEAD `3bd3c77`, but
parent attachment job `918791` exposed a fail-closed READY-field spelling mismatch in the contract
helper. The one-line correction changes the contract bundle, so `918790` and audits
`918788`/`918789` are retained only as prior-bundle evidence; a new validation and strict audit
chain is required before another attachment submission.

After the canonical READY key was frozen in commit `b5147c6`, validation `918792` passed with zero
failures on `CPU` node `nodecpu11`; the controller reported `COMPLETED/0:0` after 12 seconds. It
authorizes the strict current-bundle audit chain `918793` → `918794`. A separate validation follows
their registry update before downstream attachment work.

Post-registration validation `918795` passed with zero failures on HEAD `8f97ead`, and parent
attachment job `918796` completed. PDF child jobs `918797`–`918803` then reproducibly exited `139`
without terminal artifacts. Enabling faulthandler changes the registered Slurm job bundle, so
`918795` and audits `918793`/`918794` are retained as prior-bundle evidence pending a fresh chain.

With faulthandler enabled only in the PDF child payload, control validation `918804` passed on HEAD
`60acf0d` with zero failures; the controller reported `COMPLETED/0:0`. It authorizes strict audits
`918805` → `918806`; a separate post-registration validation precedes the diagnostic reproduction.

Post-registration validation `918807` passed and parent `918808` completed, but instrumented child
`918809` localized exit `139` to native `fitz` module creation in its later `extract` process,
before that process re-entered the full parent-binding validation, snapshot, or rendering path. Its
separate contract helper had already completed. The lazy-import and strict-import-probe correction
changes the contract bundle, so `918807`, audits `918805`/`918806`, and parent `918808` remain
prior-bundle evidence pending a fresh control and audit chain.

After lazy loading and the strict `fitz` probe were frozen in commit `070c3e5`, validation `918814`
passed with zero failures on `CPU` node `nodecpu11`. It authorizes strict audits
`918815` → `918816`; their registry update is followed by another control validation before any
new parent or CPU renderer.

Post-registration validation `918817` passed with zero failures on HEAD `8fb184f`, and parent
`918818` completed. Lazy child `918819` nevertheless reproduced exit `139` on `CPU` node
`nodecpu11`. Unlike `918809`, this child completed its pure-Python parent binding and created its
job-private PDF snapshot before the later `extract` process crashed while creating the native
PyMuPDF module. The same frozen environment imported that module successfully in L40S audit
`918816`. Binding the renderer to the registered `L40S` profile changes the contract, Slurm job,
and environment-registry bundles, so `918815`/`918816`, `918817`, and `918818` are retained as
prior-bundle evidence pending a new validation and strict audit chain.

After the L40S renderer route was frozen in commit `76ef67f`, validation `918821` completed on
`CPU` node `nodecpu10` with `status=passed`, `failure_count=0`, and exit `0`. It authorized strict
audits `918822` (`eeg2025`/CPU) → `918823` (`icml`/L40S); both completed with provenance, unchanged
locks, and no compatibility failure. Their registry update requires a separate control validation
before attachment work consumes them.

Post-registration validation `918824` passed with zero failures on HEAD `e3ed1a6`. Static review
then identified two fail-closed gaps in downstream attachment evidence validation: conflicting
non-null dependency copies were not rejected independently, and duplicated post-submit resource
fields were not compared field-by-field. No parent was submitted. Closing those gaps changes the
contract bundle, so `918822`/`918823` and `918824` remain prior-bundle evidence pending an exact
validation and audit rerun.

After dependency and post-submit hardening was frozen in commit `190285d`, validation `918825`
passed with zero failures on `CPU` node `nodecpu10`. It authorized strict audits `918826`
(`eeg2025`/CPU) → `918827` (`icml`/L40S); both completed with provenance, unchanged locks, and no
compatibility failure. A separate post-registration validation remains required.

Post-registration validation `918828` passed on HEAD `f418fb8`, and parent `918829` completed.
Top-level L40S child `918830` then failed closed with exit `3`/`OSError` during parent evidence
revalidation before its own snapshot; the single unchanged retry `918831` reproduced on node39.
Neither produced renderer output or a COMPLETE marker. Adding bounded mismatch-field/trace-location
diagnostics changes the contract bundle, so `918826`/`918827`, `918828`, and `918829` are retained
as prior-bundle evidence pending a fresh chain.
