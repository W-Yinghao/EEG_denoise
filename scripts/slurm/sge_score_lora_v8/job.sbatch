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
run="$ROOT/results/cgdr/sge_score_lora_v8/runs/${stage}/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
  tests)
    "$CPU" -m pytest -q tests/unit/test_sge_score_lora_v8.py tests/unit/test_cgdr_artifact_subspace_diffusion.py | tee "$run/pytest.txt" ;;
  eb-headroom)
    exec "$CPU" -m eeg_cgdr.cli.sge_score_lora_v8 --config configs/cgdr/sge_score_lora_v8.yaml --stage eb-headroom --run-dir "$run" ;;
  prepare-base|base-gate|support-gate|aggregate|finalize)
    exec "$CPU" -m eeg_cgdr.cli.sge_score_lora_v8 --config configs/cgdr/sge_score_lora_v8.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  base-validity|support-lora|full-eval)
    exec "$GPU" -m eeg_cgdr.cli.sge_score_lora_v8 --config configs/cgdr/sge_score_lora_v8.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  *) exit 2 ;;
esac
