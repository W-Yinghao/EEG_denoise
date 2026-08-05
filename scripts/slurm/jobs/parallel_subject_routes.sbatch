#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_parallel_explore
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml

stage=${1:?stage is required}
config=${2:-configs/cgdr/parallel_subject_aware_routes_v1.yaml}
profile=${DENOISENET_PROFILE:?DENOISENET_PROFILE is required}
case "$stage" in
    j0|fir-cache)
        python_bin="$CPU_ENV/bin/python"
        ;;
    technical|train-p0-worker|screen-base-worker|screen-dependent-worker)
        python_bin="$GPU_ENV/bin/python"
        ;;
    *)
        printf 'unsupported parallel route stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_GIT_HEAD
DENOISENET_GIT_HEAD=$(git rev-parse HEAD)
run_dir="$CODE_ROOT/results/cgdr/parallel_subject_aware_routes_v1/runs/${stage}/job_${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$run_dir"
if [[ "$stage" == j0 ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_parallel_subject_routes.py \
        tests/unit/test_cgdr_artifact_latent_diffusion.py \
        tests/unit/test_cgdr_subject_artifact_data.py \
        | tee "$run_dir/pytest.txt"
fi
exec "$python_bin" -m eeg_cgdr.cli.parallel_routes \
    --config "$CODE_ROOT/$config" \
    --stage "$stage" \
    --run-dir "$run_dir"
