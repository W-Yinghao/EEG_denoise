#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly DATA_ROOT=/projects/EEG-foundation-model
readonly REPORT_ROOT="$CODE_ROOT/reports/benchmark_data_locator"

[[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
    printf 'benchmark data locator requires the cpu profile\n' >&2
    exit 2
}
[[ $# -eq 0 ]] || {
    printf 'benchmark data locator accepts no arguments\n' >&2
    exit 2
}
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || {
    printf 'data root is unavailable or is a symbolic link\n' >&2
    exit 1
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
run_dir="$REPORT_ROOT/$job_id"
mkdir -p -- "$run_dir"
matches="$run_dir/matches.txt"
status="$run_dir/status.txt"

# This is deliberately a name-only recursive lookup for the exact files and
# directory labels used by the frozen official benchmark sources. It never
# opens signal payloads and never computes checksums or sizes.
set +e
timeout 900 find "$DATA_ROOT" -xdev \
    \( -type f \( \
        -iname 'EEG_all_epochs.npy' -o \
        -iname 'EOG_all_epochs.npy' -o \
        -iname 'EMG_all_epochs.npy' -o \
        -iname 'ECG_all_epochs.npy' -o \
        -iname 'ssed_noise.npy' -o \
        -iname 'ssed_eeg.npy' \
    \) -o -type d \( \
        -iname 'EEGDfus' -o \
        -iname 'D4PM' -o \
        -iname 'SSED' -o \
        -iname 'EEGdenoiseNet' \
    \) \) -print > "$matches"
find_rc=$?
set -e

case "$find_rc" in
    0) state=complete_recursive_name_lookup ;;
    124) state=incomplete_timeout_no_absence_claim ;;
    *) state=failed_recursive_name_lookup ;;
esac
printf '%s|%s|%s|%s\n' "$job_id" "$state" "$find_rc" "$DATA_ROOT" > "$status"
[[ "$find_rc" -eq 0 ]]

