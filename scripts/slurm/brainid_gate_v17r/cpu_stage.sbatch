#!/usr/bin/env bash
#SBATCH --job-name=brainid17r_cpu
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%A_%a.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17r
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo";export PYTHONPATH="$repo/src";mkdir -p "$repo/.slurm/brainid_gate_v17r"
stage="${STAGE:?}";task="${SLURM_ARRAY_TASK_ID:-0}";run="$repo/results/cgdr/brainid_gate_v17r/server_runs/$stage/task_$task"
python -m eeg_cgdr.cli.brainid_gate_v17r --config configs/cgdr/brainid_gate_v17r.yaml --stage "$stage" --index "$task" --run-dir "$run"
