#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_raw_support_clean_diffusion
CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
jid=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
run="$ROOT/results/cgdr/raw_support_clean_diffusion/runs/$stage/job_${jid}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
 tests) "$CPU" -m pytest -q tests/unit/test_raw_support_clean_diffusion.py | tee "$run/pytest.txt" ;;
 clean-import) tmp=${SLURM_TMPDIR:-$run/clean}; mkdir -p "$tmp"; git archive HEAD | tar -x -C "$tmp"; PYTHONPATH="$tmp/src:$tmp" "$CPU" -c 'import eeg_cgdr.experiments.raw_support_clean_diffusion; import eeg_cgdr.models.raw_support_clean_diffusion; print("clean import passed")' ;;
 audit|geometry-support|geometry-evaluate|geometry-aggregate|oracle-aggregate|materialize|evaluate|evaluate-additional|aggregate|final) "$CPU" -m eeg_cgdr.cli.raw_support_clean_diffusion --config configs/cgdr/raw_support_clean_diffusion.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
 *) "$GPU" -m eeg_cgdr.cli.raw_support_clean_diffusion --config configs/cgdr/raw_support_clean_diffusion.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
esac
