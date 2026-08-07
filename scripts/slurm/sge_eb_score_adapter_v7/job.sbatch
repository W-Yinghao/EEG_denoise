#!/usr/bin/env bash
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_sge_transfer_v6
readonly CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
readonly GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
stage=${1:?stage};shift
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
run="$ROOT/results/cgdr/sge_dynamic_transfer_diffusion_v6/runs/v7-${stage}/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
  tests)
    "$CPU" -m pytest -q tests/unit/test_sge_eb_score_adapter_v7.py tests/unit/test_sge_dynamic_transfer_v6.py | tee "$run/pytest.txt" ;;
  cpu-audit)
    exec "$CPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage cpu-audit --run-dir "$run" ;;
  fold-audit)
    exec "$GPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage fold-audit --task-index "${SLURM_ARRAY_TASK_ID:?}" --run-dir "$run" ;;
  overfit)
    exec "$GPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage overfit --task-index "${SLURM_ARRAY_TASK_ID:?}" --run-dir "$run" ;;
  validity-decision)
    exec "$CPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage validity-decision --overfit-job "${V7_OVERFIT_JOB:?}" --run-dir "$run" ;;
  eb-headroom)
    exec "$CPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage eb-headroom --run-dir "$run" ;;
  finalize)
    exec "$CPU" -m eeg_cgdr.cli.sge_eb_score_adapter_v7 --config configs/cgdr/sge_eb_score_adapter_v7.yaml --stage finalize --run-dir "$run" ;;
  *) exit 2 ;;
esac
