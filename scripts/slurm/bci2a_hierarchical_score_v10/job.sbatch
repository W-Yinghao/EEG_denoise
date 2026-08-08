#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_bci2a_v10
CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
stage=${1:?stage}; cd "$ROOT"; export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"; export CUBLAS_WORKSPACE_CONFIG=:4096:8
array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}; run="$ROOT/results/cgdr/bci2a_hierarchical_score_v10/runs/${stage}/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"; mkdir -p "$run"
case "$stage" in
  tests) "$CPU" -m pytest -q tests/unit/test_bci2a_hierarchical_score_v10.py | tee "$run/pytest.txt" ;;
  clean-import) clean_root=${SLURM_TMPDIR:-$run/clean-checkout}; mkdir -p "$clean_root"; git archive HEAD | tar -x -C "$clean_root"; PYTHONPATH="$clean_root/src:$clean_root" "$CPU" -c 'import eeg_cgdr.data.bci2a_v10; import eeg_cgdr.experiments.bci2a_hierarchical_score_v10; print("V10 clean import passed")' ;;
  inventory|sge-closure|sge-factorial-eval|headroom-extract|headroom-aggregate|bci2b-extract|bci2b-aggregate|prepare-fold|eval-fold|aggregate|diagnostic-aggregate|finalize) exec "$CPU" -m eeg_cgdr.cli.bci2a_hierarchical_score_v10 --config configs/cgdr/bci2a_hierarchical_score_v10.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  sge-factorial-replay|train-fold|infer-fold) exec "$GPU" -m eeg_cgdr.cli.bci2a_hierarchical_score_v10 --config configs/cgdr/bci2a_hierarchical_score_v10.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  *) exit 2 ;;
esac
