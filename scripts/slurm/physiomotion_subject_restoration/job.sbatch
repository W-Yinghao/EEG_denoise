#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run="$ROOT/results/cgdr/physiomotion_subject_restoration/runs/$stage/job_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+/task_$SLURM_ARRAY_TASK_ID}"
mkdir -p "$run"
case "$stage" in
 tests) "$PY" -m pytest -q tests/unit/test_physiomotion_subject_restoration.py ;;
 clean-import) tmp=${SLURM_TMPDIR:-/tmp}/pm_${SLURM_JOB_ID}; mkdir -p "$tmp"; git archive HEAD | tar -x -C "$tmp"; PYTHONPATH="$tmp/src:$tmp" "$PY" -c 'import eeg_cgdr.experiments.physiomotion_subject_restoration' ;;
 *) "$PY" -m eeg_cgdr.cli.physiomotion_subject_restoration --config configs/cgdr/physiomotion_subject_restoration.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
esac
