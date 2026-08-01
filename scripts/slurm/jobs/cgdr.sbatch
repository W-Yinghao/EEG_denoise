#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
mode=${1:-}
config=${2:-configs/cgdr/klados_v4.yaml}
stage=${3:-}
[[ $# -le 3 ]] || { printf 'usage: cgdr.sbatch MODE [CONFIG] [STAGE]\n' >&2; exit 2; }
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
    mechanism-audit)
        case "$stage" in
            ""|legacy-direction-check|cpu-tests)
                [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
                    printf 'mechanism-audit %s requires cpu\n' "${stage:-legacy}" >&2; exit 2;
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            aggregate-development|decision)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'mechanism-audit %s requires cpu-high\n' "$stage" >&2; exit 2;
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            sampler-integration)
                [[ "${DENOISENET_PROFILE:-}" == L40S || "${DENOISENET_PROFILE:-}" == A100 ]] || {
                    printf 'mechanism-audit sampler-integration requires L40S or A100\n' >&2; exit 2;
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            train-prior|development-record|untouched-record)
                [[ "${DENOISENET_PROFILE:-}" == A100 || "${DENOISENET_PROFILE:-}" == H100 ]] || {
                    printf 'mechanism-audit %s requires A100 or H100\n' "$stage" >&2; exit 2;
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            *) printf 'unknown mechanism-audit stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        ;;
    *) printf 'unknown CGDR mode: %s\n' "$mode" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT"
export HARNESS_LEVEL=1
job_id=${SLURM_JOB_ID:?}
task_suffix=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_suffix="_${SLURM_ARRAY_TASK_ID}"
fi
stage_suffix=""
if [[ -n "$stage" ]]; then
    stage_suffix="_${stage}"
fi
run_dir="$CODE_ROOT/runs/cgdr_${mode}${stage_suffix}_${job_id}${task_suffix}"
mkdir -p "$run_dir"
cp -- "$config" "$run_dir/config.yaml"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s\n' "$SLURM_ARRAY_TASK_ID" > "$run_dir/slurm_array_task_id.txt"
fi

if [[ "$mode" == mechanism-audit && "$stage" == cpu-tests ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_p0_precision.py \
        tests/unit/test_cgdr_sampler_semantics.py \
        tests/unit/test_cgdr_mechanism_data.py \
        tests/unit/test_cgdr_mechanism_aggregate.py \
        tests/unit/test_cgdr_mechanism_runner.py
fi

child_pid=""
forward_usr1() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -USR1 "$child_pid"
    fi
}
trap forward_usr1 USR1

cli_args=("$mode" --config "$config" --run-dir "$run_dir")
if [[ -n "$stage" ]]; then
    cli_args+=(--stage "$stage")
fi
"$python_bin" -m eeg_cgdr.cli.main "${cli_args[@]}" &
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
