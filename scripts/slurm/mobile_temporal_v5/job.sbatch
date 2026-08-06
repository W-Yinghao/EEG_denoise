#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_mobile_diffusion_v5
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
stage=${1:?stage required}
case "$stage" in
  metadata|preprocess|protocol|aggregate|aggregate-three-seed|pc-selector|report|tests) python_bin="$CPU_ENV/bin/python" ;;
  technical|factorial|additional) python_bin="$GPU_ENV/bin/python" ;;
  *) printf 'unsupported stage: %s\n' "$stage" >&2; exit 2 ;;
esac
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export DENOISENET_SGE_STRUCTURE_AUDIT=/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919218/attempt-0/result.json
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_dir="$CODE_ROOT/results/cgdr/mobile_temporal_diffusion_v5/runs/$stage/job_${SLURM_JOB_ID}"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_dir"
/usr/bin/git rev-parse HEAD > "$run_dir/git_head.txt"
if [[ "$stage" == tests ]]; then
  "$python_bin" -m pytest -q tests/unit/test_mobile_temporal_v5.py tests/unit/test_temporal_support_v4.py tests/unit/test_pc_constrained_oracle.py | tee "$run_dir/pytest.txt"
fi
exec "$python_bin" -m eeg_cgdr.cli.mobile_temporal_v5 --config "$CODE_ROOT/configs/cgdr/mobile_temporal_diffusion_v5/protocol.yaml" --stage "$stage" --run-dir "$run_dir"
