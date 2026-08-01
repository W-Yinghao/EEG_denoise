# Registered environment audit

Both initial audits completed through Slurm. No environment was created, upgraded, or modified. A subsequent control-plane review found that those jobs did not strictly fail on every critical import/GPU probe, retained unsanitized lock streams, and used a shared `latest` pointer. They are therefore preserved as initial observations but are not accepted as final compatibility evidence; strict re-audits are pending.

| Environment | Slurm job / allocation | Python | Key verified capability | Explicit lock SHA-256 | Status |
|---|---|---|---|---|---|
| `eeg2025` | `918736`; `CPU`; node `nodecpu10`; 2 CPU; 8 GiB | 3.13.7 | Initial observation: NumPy 2.4.4, SciPy 1.17.0, MNE 1.11.0, h5py 3.15.1 | `cc644eead3ffa906a70573727b82974da57145b408b67a920741a4288fc0296d` | provenance incomplete; strict re-audit pending |
| `icml` | `918737`; `L40S`; node `node39`; 8 CPU; 64 GiB; 1 GPU | 3.9.25 | Initial observation: PyTorch 2.8.0+cu128, CUDA available, cuDNN 91002, one NVIDIA L40S visible | `2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1` | provenance incomplete; strict re-audit pending |

The CPU audit correctly saw no CUDA device; its installed PyTorch build is not used as evidence of GPU compatibility. The `icml` allocation reported NVIDIA driver 595.58.03 and 46,068 MiB total device memory. Full explicit Conda manifests, `pip freeze`, import results, Slurm allocation snapshots, and command exit codes are retained under:

- `reports/environments/eeg2025/jobs/918736/`
- `reports/environments/icml/jobs/918737/`

Because `sacct` is unavailable, initial completion is evidenced by each payload's atomic `status.json` plus the live `scontrol show job` snapshot captured during execution. Historical accounting fields that were not observable are not inferred. Final responsibility/compatibility status will be assigned only from the strict re-audit artifacts.
