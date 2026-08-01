#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly MANIFEST="$CODE_ROOT/reports/dataset_harness/jobs/919275/attempt-0/result.json"
umask 077
cd "$CODE_ROOT"

payload_mode=${1:-pilot}
[[ $# -le 1 && "$payload_mode" =~ ^(pilot|remaining-shards)$ ]] || {
    printf 'eye_bci_download accepts only pilot or remaining-shards\n' >&2
    exit 2
}
[[ "${DENOISENET_PROFILE:-}" == cpu-high && "${SLURM_JOB_PARTITION:-}" == cpu-high ]] || {
    printf 'Eye-BCI download requires the cpu-high profile\n' >&2
    exit 2
}
array_job_id=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
array_task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
attempt=${SLURM_RESTART_COUNT:-0}
run_dir="$CODE_ROOT/reports/eye_bci_download/jobs/$array_job_id/task-$array_task_id/attempt-$attempt"
mkdir -p -- "$(dirname "$run_dir")"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 1; }
mkdir -- "$run_dir"

started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
finish() {
    rc=$?
    trap - EXIT
    finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then state=completed; else state=failed; fi
    printf '{\n  "array_job_id": "%s",\n  "array_task_id": %d,\n  "state": "%s",\n  "exit_code": %d,\n  "started_at_utc": "%s",\n  "finished_at_utc": "%s"\n}\n' \
        "$array_job_id" "$array_task_id" "$state" "$rc" "$started_at" "$finished_at" \
        > "$run_dir/status.json.partial"
    mv -- "$run_dir/status.json.partial" "$run_dir/status.json"
    exit "$rc"
}
trap finish EXIT

if [[ "$payload_mode" == pilot ]]; then
    "$EEG_ENV/bin/python" "$CODE_ROOT/scripts/eye_bci_download.py" subject \
        --manifest "$MANIFEST" \
        --array-index "$array_task_id" \
        --output "$run_dir/result.json"
else
    shard_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
    "$EEG_ENV/bin/python" "$CODE_ROOT/scripts/eye_bci_download.py" shard \
        --manifest "$MANIFEST" \
        --array-index "$array_task_id" \
        --shard-count "$shard_count" \
        --output "$run_dir/result.json"
fi
