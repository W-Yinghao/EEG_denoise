#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly MANIFEST="$CODE_ROOT/reports/dataset_harness/jobs/919275/attempt-0/result.json"
umask 077
cd "$CODE_ROOT"

[[ $# -eq 0 ]] || { printf 'eye_bci_finalize accepts no payload arguments\n' >&2; exit 2; }
[[ "${DENOISENET_PROFILE:-}" == cpu && "${SLURM_JOB_PARTITION:-}" == CPU ]] || {
    printf 'Eye-BCI finalizer requires the cpu profile\n' >&2
    exit 2
}
job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
attempt=${SLURM_RESTART_COUNT:-0}
run_dir="$CODE_ROOT/reports/eye_bci_download/finalize/$job_id/attempt-$attempt"
mkdir -p -- "$(dirname "$run_dir")"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 1; }
mkdir -- "$run_dir"

started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
finish() {
    rc=$?
    trap - EXIT
    finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then state=completed; else state=failed; fi
    printf '{\n  "job_id": "%s",\n  "state": "%s",\n  "exit_code": %d,\n  "started_at_utc": "%s",\n  "finished_at_utc": "%s"\n}\n' \
        "$job_id" "$state" "$rc" "$started_at" "$finished_at" \
        > "$run_dir/status.json.partial"
    mv -- "$run_dir/status.json.partial" "$run_dir/status.json"
    exit "$rc"
}
trap finish EXIT

"$EEG_ENV/bin/python" "$CODE_ROOT/scripts/eye_bci_download.py" finalize \
    --manifest "$MANIFEST" \
    --output "$run_dir/result.json"
