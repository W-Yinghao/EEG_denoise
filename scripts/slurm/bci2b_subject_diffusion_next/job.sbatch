#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_subject_diffusion_next
CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
jid=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
run="$ROOT/results/cgdr/bci2b_subject_diffusion_next_runs/$stage/job_${jid}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run"
case "$stage" in
 tests) "$CPU" -m pytest -q tests/unit/test_bci2b_subject_diffusion_next.py | tee "$run/pytest.txt" ;;
 clean-import) d=${SLURM_TMPDIR:-$run/clean};mkdir -p "$d";git archive HEAD|tar -x -C "$d";PYTHONPATH="$d/src:$d" "$CPU" -c 'import eeg_cgdr.experiments.bci2b_subject_diffusion_next; print("clean import passed")' ;;
 audit|prepare-pop8|prepare-pop8-all|causal-audit|causal-audit-all|aggregate-causal|evaluate|evaluate-all|evaluate-oppost-all|aggregate-oppost|evaluate-local-all|aggregate-local|aggregate|finalize) "$CPU" -m eeg_cgdr.cli.bci2b_subject_diffusion_next --config configs/cgdr/bci2b_subject_diffusion_next.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
 *) "$GPU" -m eeg_cgdr.cli.bci2b_subject_diffusion_next --config configs/cgdr/bci2b_subject_diffusion_next.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
esac
