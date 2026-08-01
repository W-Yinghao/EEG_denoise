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
the validator that authorizes the current-bundle strict environment audits `918774`/`918775`.
