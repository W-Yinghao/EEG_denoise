# Slurm job ledger

Snapshot scope: all 94 numeric submission manifests present under `reports/slurm/submissions/` at the 2026-08-01 control-plane review. The row-complete machine ledger is `reports/slurm/job_ledger.csv`.

## Evidence rules

- Requested job, profile, partition, CPU, memory, GRES, wall time, account, QOS and dependency fields come from each immutable submission manifest. All 94 manifest `array` fields are empty, so no array submission is present and no repeated empty array column is added to the CSV.
- State and exit code come from the corresponding no-replace payload `status.json` where one exists. A missing terminal marker is reported as `unavailable`, not inferred as success.
- Node and `AllocTRES` come only from the corresponding live `slurm_allocation.json` captured while the allocation was running. These are start-time allocation snapshots, not final Slurm accounting records.
- `sacct` could not connect to SlurmDBD (`Connection refused`). Final scheduler elapsed/end accounting, energy, peak memory and historical allocation fields are therefore unavailable unless separately captured by a payload; none is invented here.
- Satisfied dependencies may appear as `Dependency=(null)` in a later live controller view. The CSV retains the original dependency from the submission manifest.
- No listed job is a scientific training, P0, gate or outer-test run. Successful rows are administrative evidence only; failures and cancellations are failure evidence only.

## Failure and recovery chronology

- Jobs 918736/918737 and 918740/918742 are retained as first-generation environment and attachment observations with incomplete strict provenance. Their missing live allocation JSON fields remain `unavailable`.
- The NFS-safe attachment and READY-contract iterations are retained across 918768–918795. Parent jobs 918777 and 918791 failed closed; the later parent 918796 completed its bounded parent phase.
- CPU PyMuPDF children 918797–918803, 918809 and 918819 exited 139. The record retains every sibling and the single unchanged retry rather than presenting only later infrastructure recoveries.
- L40S parent-revalidation diagnostics 918830/918831 and 918840 exited 3. Cold-start renderer children 918847/918848 exited 139. Diagnostic GPU audits 918851, 918855 and 918888 failed their preregistered renderer policies.
- Job 918885 is the invalid CPU audit invocation whose forbidden `afterok:918884` dependency was rejected before runtime probing. Job 918886 depended on 918885 and was cancelled as `DependencyNeverSatisfied`; it received no allocation. Corrected CPU audit 918887 removed the dependency.
- The first shared-startup chain was 918899 → 918900 → 918901 → 918902 → 918903 → 918904. Job 918904 failed closed during PDF extraction and did not authorize sibling PDF jobs.
- The bounded-failure chain was 918907 → 918908 → 918909 → 918915 → 918916 → 918917. Current environment audits 918908/918909 and verified-state control 918915 completed, parent 918916 completed its parent phase, and PDF canary 918917 failed closed at the page-8 text warning audit.
- Inventory job 918918 was still `PENDING (Priority)` in the captured controller snapshot. Its original dependency is `afterok:918908`; the scheduler displayed an estimated start of 2026-08-05T00:40:00. That time and `SchedNodeList` are scheduling estimates, not an allocation. The ledger therefore records no node or `AllocTRES` for 918918 and must be updated from its own immutable status/allocation evidence after it starts or terminates.

## Scientific boundary

The ledger proves submission and bounded administrative execution history only. It does not prove a verified real EEG dataset, frozen split, population posterior, NULL equivalence, P0 slice, G1–G5 result, or B1–B6 diagnostic. Legacy repository jobs and results outside these 94 contract submissions are not promoted into this ledger as current scientific evidence.
