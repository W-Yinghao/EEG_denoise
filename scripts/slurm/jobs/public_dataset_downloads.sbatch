#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly CONFIG="$CODE_ROOT/configs/dataset_sources.yaml"
umask 077
cd "$CODE_ROOT"

mode=${1:-}
case "$mode" in
    self-test|download-klados|plan-klados-v4|download-klados-v4|plan-sgeyesub|download-sgeyesub) ;;
    *) printf 'usage: public_dataset_downloads.sbatch {self-test|download-klados|plan-klados-v4|download-klados-v4|plan-sgeyesub|download-sgeyesub}\n' >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { printf 'unexpected public download arguments\n' >&2; exit 2; }
[[ "${DENOISENET_PROFILE:-}" == cpu && "${SLURM_JOB_PARTITION:-}" == CPU ]] || {
    printf 'public downloads require the cpu profile\n' >&2
    exit 2
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
attempt=${SLURM_RESTART_COUNT:-0}
run_dir="$CODE_ROOT/reports/dataset_downloads/jobs/$job_id/attempt-$attempt"
mkdir -p -- "$CODE_ROOT/reports/dataset_downloads/jobs/$job_id"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 1; }
mkdir -- "$run_dir"

started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
finish() {
    rc=$?
    trap - EXIT
    finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then state=completed; else state=failed; fi
    status_tmp="$run_dir/status.json.partial"
    printf '{\n  "job_id": "%s",\n  "attempt": %d,\n  "mode": "%s",\n  "state": "%s",\n  "exit_code": %d,\n  "started_at_utc": "%s",\n  "finished_at_utc": "%s"\n}\n' \
        "$job_id" "$attempt" "$mode" "$state" "$rc" "$started_at" "$finished_at" > "$status_tmp"
    mv -- "$status_tmp" "$run_dir/status.json"
    exit "$rc"
}
trap finish EXIT

"$EEG_ENV/bin/python" "$CODE_ROOT/scripts/public_dataset_downloads.py" "$mode" \
    --config "$CONFIG" \
    --output "$run_dir/result.json"
