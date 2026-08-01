#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
mode=${1:-}
config=${2:-configs/cgdr/klados_v4.yaml}
[[ $# -le 2 ]] || { printf 'usage: cgdr.sbatch MODE [CONFIG]\n' >&2; exit 2; }
case "$mode" in
    metadata|cpu-validate)
        [[ "${DENOISENET_PROFILE:-}" == cpu ]] || { printf '%s requires cpu\n' "$mode" >&2; exit 2; }
        python_bin="$EEG_ENV/bin/python"
        ;;
    gpu-integrate)
        [[ "${DENOISENET_PROFILE:-}" == L40S || "${DENOISENET_PROFILE:-}" == A100 ]] || {
            printf 'gpu-integrate requires L40S or A100\n' >&2; exit 2;
        }
        python_bin="$GPU_ENV/bin/python"
        ;;
    train-fold|eye-fold)
        [[ "${DENOISENET_PROFILE:-}" == L40S \
            || "${DENOISENET_PROFILE:-}" == A100 \
            || "${DENOISENET_PROFILE:-}" == H100 \
            || "${DENOISENET_PROFILE:-}" == V100-32GB ]] || {
            printf '%s requires a registered GPU profile\n' "$mode" >&2; exit 2;
        }
        python_bin="$GPU_ENV/bin/python"
        ;;
    *) printf 'unknown CGDR mode: %s\n' "$mode" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT"
export HARNESS_LEVEL=1
job_id=${SLURM_JOB_ID:?}
run_dir="$CODE_ROOT/runs/cgdr_${mode}_${job_id}"
mkdir -p "$run_dir"
cp -- "$config" "$run_dir/config.yaml"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"

child_pid=""
forward_usr1() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -USR1 "$child_pid"
    fi
}
trap forward_usr1 USR1

"$python_bin" -m eeg_cgdr.cli.main "$mode" --config "$config" --run-dir "$run_dir" &
child_pid=$!
while true; do
    set +e
    wait "$child_pid"
    child_status=$?
    set -e
    if kill -0 "$child_pid" 2>/dev/null; then
        continue
    fi
    exit "$child_status"
done
