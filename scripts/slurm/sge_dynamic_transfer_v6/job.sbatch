#!/usr/bin/env bash
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_sge_transfer_v6
readonly CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
readonly GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
stage=${1:?stage};shift
cd "$ROOT";export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run="$ROOT/results/cgdr/sge_dynamic_transfer_diffusion_v6/runs/$stage/job_${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}";mkdir -p "$run"
case "$stage" in
  audit) exec "$CPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage audit --run-dir "$run";;
  build) exec "$CPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage build --task-index "${SLURM_ARRAY_TASK_ID:?}" --run-dir "$run";;
  technical) exec "$GPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage technical --run-dir "$run";;
  train-det|train-diff) kind=${stage#train-};task=$((SLURM_ARRAY_TASK_ID/2));slot=$((SLURM_ARRAY_TASK_ID%2));[[ "$kind" == det && "$slot" == 0 || "$kind" == diff && "$slot" == 1 ]] || exit 0;exec "$GPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage train --task-index "$task" --model-kind "$kind" --seed "${V6_SEED:-20260806}" --run-dir "$run";;
  train) task=$((SLURM_ARRAY_TASK_ID/2));kind=det;((SLURM_ARRAY_TASK_ID%2))&&kind=diff;exec "$GPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage train --task-index "$task" --model-kind "$kind" --seed "${V6_SEED:-20260806}" --run-dir "$run";;
  train-both) task=${SLURM_ARRAY_TASK_ID:?};"$GPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage train --task-index "$task" --model-kind det --seed "${V6_SEED:-20260806}" --run-dir "$run/det";exec "$GPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage train --task-index "$task" --model-kind diff --seed "${V6_SEED:-20260806}" --run-dir "$run/diff";;
  aggregate) exec "$CPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage aggregate --seed "${V6_SEED:-20260806}" --run-dir "$run";;
  report) exec "$CPU" -m eeg_cgdr.cli.sge_dynamic_transfer_v6 --config configs/cgdr/sge_dynamic_transfer_diffusion_v6.yaml --stage report --run-dir "$run";;
  tests) "$CPU" -m pytest -q tests/unit/test_sge_dynamic_transfer_v6.py tests/unit/test_mobile_temporal_v5.py | tee "$run/pytest.txt";;
  mobile-closure) exec "$CPU" -m eeg_cgdr.cli.mobile_v5_closure --run-dir "$run";;
  *) exit 2;;
esac
