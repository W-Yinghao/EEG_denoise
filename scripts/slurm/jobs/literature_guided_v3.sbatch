#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_lit_explore_v3
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml

stage=${1:?stage is required}
config=${2:-configs/cgdr/literature_guided_exploration_v3/exploration.yaml}
profile=${DENOISENET_PROFILE:?DENOISENET_PROFILE is required}
case "$stage" in
    j0-audit|j1-tasks|j7-aggregate|j8-final)
        python_bin="$CPU_ENV/bin/python"
        ;;
    j2-tech|j2-baselines|j3-raw-support|j4-lora|j5-selector|j6-control)
        python_bin="$GPU_ENV/bin/python"
        ;;
    *)
        printf 'unsupported literature-v3 stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_GIT_HEAD
DENOISENET_GIT_HEAD=$(git rev-parse HEAD)
run_dir="$CODE_ROOT/results/cgdr/literature_guided_exploration_v3/runs/${stage}/job_${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$run_dir"

if [[ "$stage" == j0-audit || "$stage" == j1-tasks || "$stage" == j8-final ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_literature_guided_v3.py \
        | tee "$run_dir/pytest.txt"
fi

exec "$python_bin" -m eeg_cgdr.cli.literature_guided_v3 \
    --config "$CODE_ROOT/$config" \
    --stage "$stage" \
    --run-dir "$run_dir"
