#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_score_lora_v8
readonly CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
readonly GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
stage=${1:?stage}
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
run="$ROOT/results/cgdr/sge_eb_bridge_v8_1/runs/${stage}/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
  tests)
    "$CPU" -m pytest -q tests/unit/test_sge_eb_bridge_v8_1.py tests/unit/test_sge_score_lora_v8.py tests/unit/test_cgdr_artifact_subspace_diffusion.py | tee "$run/pytest.txt" ;;
  prepare-new|eb-robustness|eb-resume|aggregate|support-inner-gate|finalize)
    exec "$CPU" -m eeg_cgdr.cli.sge_eb_bridge_v8_1 --config configs/cgdr/sge_eb_bridge_v8_1.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  train-new|validity-core|repaired-inference|bridge|support-inner)
    exec "$GPU" -m eeg_cgdr.cli.sge_eb_bridge_v8_1 --config configs/cgdr/sge_eb_bridge_v8_1.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  *) exit 2 ;;
esac
