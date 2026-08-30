#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EXTERNAL_ROOT="$CODE_ROOT/.external"
readonly REPORT_ROOT="$CODE_ROOT/reports/benchmark_source_checkout"

[[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
    printf 'benchmark source checkout requires the cpu profile\n' >&2
    exit 2
}

benchmark=${1:-}
case "$benchmark" in
    eegdfus)
        url=https://github.com/XYH0118/EEGDfus.git
        directory=EEGDfus
        ;;
    d4pm)
        url=https://github.com/flysnow1024/D4PM.git
        directory=D4PM
        ;;
    ds-ddpm)
        url=https://github.com/duanyiqun/DS-DDPM.git
        directory=DS-DDPM
        ;;
    *)
        printf 'usage: benchmark_source_checkout.sbatch {eegdfus|d4pm|ds-ddpm}\n' >&2
        exit 2
        ;;
esac
[[ $# -eq 1 ]] || {
    printf 'unexpected benchmark source checkout arguments\n' >&2
    exit 2
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
destination="$EXTERNAL_ROOT/$directory"
partial="$EXTERNAL_ROOT/.${directory}.partial-$job_id"
report_dir="$REPORT_ROOT/$job_id"
mkdir -p -- "$EXTERNAL_ROOT" "$report_dir"

if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -d "$destination/.git" && ! -L "$destination" ]] || {
        printf 'existing destination is not a safe Git checkout: %s\n' "$destination" >&2
        exit 1
    }
    actual_url=$(git -C "$destination" remote get-url origin)
    [[ "$actual_url" == "$url" || "$actual_url" == "${url%.git}" ]] || {
        printf 'existing checkout has unexpected origin: %s\n' "$actual_url" >&2
        exit 1
    }
    state=already_present
else
    [[ ! -e "$partial" && ! -L "$partial" ]] || {
        printf 'partial checkout already exists: %s\n' "$partial" >&2
        exit 1
    }
    git clone --quiet --filter=blob:none --no-tags "$url" "$partial"
    if [[ -e "$destination" || -L "$destination" ]]; then
        printf 'destination appeared concurrently; retaining partial without overwrite\n' >&2
        exit 1
    fi
    mv -- "$partial" "$destination"
    state=cloned
fi

commit=$(git -C "$destination" rev-parse HEAD)
branch=$(git -C "$destination" branch --show-current)
origin=$(git -C "$destination" remote get-url origin)
printf '%s\n' "$commit" > "$report_dir/commit.txt"
printf '%s\n' "$origin" > "$report_dir/origin.txt"
printf '%s\n' "$branch" > "$report_dir/branch.txt"
printf '%s|%s|%s|%s|%s\n' "$job_id" "$benchmark" "$state" "$commit" "$destination" \
    > "$report_dir/result.txt"

