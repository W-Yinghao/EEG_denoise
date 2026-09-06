#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_bci2b_eog_residual_v11
CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
run="$ROOT/results/cgdr/bci2b_eog_residual_v11/runs/$stage/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
  tests) "$CPU" -m pytest -q tests/unit/test_bci2b_eog_residual_v11.py | tee "$run/pytest.txt" ;;
  clean-import) clean_root=${SLURM_TMPDIR:-$run/clean-checkout}; mkdir -p "$clean_root"; git archive HEAD | tar -x -C "$clean_root"; PYTHONPATH="$clean_root/src:$clean_root" "$CPU" -c 'import eeg_cgdr.experiments.bci2b_eog_residual_v11; print("V11 clean import passed")' ;;
  bridge-audit|bridge-aggregate|prepare-fold|eval-fold|gate3|aggregate|finalize) exec "$CPU" -m eeg_cgdr.cli.bci2b_eog_residual_v11 --config configs/cgdr/bci2b_eog_residual_v11.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  *) exec "$GPU" -m eeg_cgdr.cli.bci2b_eog_residual_v11 --config configs/cgdr/bci2b_eog_residual_v11.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
esac
