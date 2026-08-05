#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_parallel_explore
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml

stage=${1:?stage is required}
config=${2:-configs/cgdr/subject_bridge_repair.yaml}
case "$stage" in
    j0|aggregate) python_bin="$CPU_ENV/bin/python" ;;
    technical|train-worker|infer-worker) python_bin="$GPU_ENV/bin/python" ;;
    *) printf 'unsupported subject bridge stage: %s\n' "$stage" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export DENOISENET_SGE_STRUCTURE_AUDIT=/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919218/attempt-0/result.json
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_GIT_HEAD
DENOISENET_GIT_HEAD=$(git rev-parse HEAD)
run_dir="$CODE_ROOT/results/cgdr/subject_bridge_repair/runs/${stage}/job_${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"; fi
mkdir -p "$run_dir"
if [[ "$stage" == j0 ]]; then
    "$python_bin" -m pytest -q tests/unit/test_cgdr_subject_bridge.py | tee "$run_dir/pytest.txt"
fi
exec "$python_bin" -m eeg_cgdr.cli.subject_bridge --config "$CODE_ROOT/$config" --stage "$stage" --run-dir "$run_dir"
