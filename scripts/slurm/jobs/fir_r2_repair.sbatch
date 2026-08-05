#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_fir_r2_repair
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml

stage=${1:?repair stage is required}
config=${2:-configs/cgdr/subject_aware_diffusion_exploration_v2_fir_r2_repair.yaml}
case "$stage" in
    tests|j1-operator-audit|j4-rank|j7-aggregate)
        python_bin="$CPU_ENV/bin/python"
        ;;
    j2-technical|j3-carrier-worker)
        python_bin="$GPU_ENV/bin/python"
        ;;
    *)
        printf 'unsupported FIR R2 repair stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export DENOISENET_SGE_STRUCTURE_AUDIT=/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919218/attempt-0/result.json
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_GIT_HEAD
DENOISENET_GIT_HEAD=$(git rev-parse HEAD)
run_dir="$CODE_ROOT/results/cgdr/subject_aware_diffusion_exploration_v2_fir_r2_repair/runs/$stage/job_${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$run_dir"

if [[ "$stage" == tests ]]; then
    exec "$python_bin" -m pytest -q tests/unit/test_cgdr_subject_aware_diffusion_exploration_v2.py
fi

exec "$python_bin" -m eeg_cgdr.cli.main subject-aware-wide-v2 \
    --config "$CODE_ROOT/$config" \
    --run-dir "$run_dir" \
    --stage "$stage"
