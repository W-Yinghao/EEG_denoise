#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_mobile_headroom_v4
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
stage=${1:?stage required}
case "$stage" in
  j0-tests|j1a-v3-repair|j1a-pc-oracle|j1a-pc-selector|j1b-metadata|j2-signal-audit|j3-evaluator|j5-headroom-route|j7-final) python_bin="$CPU_ENV/bin/python" ;;
  j4-headroom|j6-temporal) python_bin="$GPU_ENV/bin/python" ;;
  *) printf 'unsupported stage: %s\n' "$stage" >&2; exit 2 ;;
esac
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_SGE_STRUCTURE_AUDIT=/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919218/attempt-0/result.json
run_dir="$CODE_ROOT/results/cgdr/mobile_bci_headroom_v4/runs/$stage/job_${SLURM_JOB_ID}"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_dir"
if [[ "$stage" == j0-tests || "$stage" == j7-final ]]; then
  "$python_bin" -m pytest -q tests/unit/test_mobile_bci_v4.py tests/unit/test_pc_constrained_oracle.py tests/unit/test_temporal_support_v4.py | tee "$run_dir/pytest.txt"
fi
exec "$python_bin" -m eeg_cgdr.cli.mobile_bci_headroom_v4 \
  --config "$CODE_ROOT/configs/cgdr/mobile_bci_headroom_v4/protocol.yaml" --stage "$stage" --run-dir "$run_dir"
