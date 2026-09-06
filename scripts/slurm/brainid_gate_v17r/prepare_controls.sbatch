#!/usr/bin/env bash
#SBATCH --job-name=brainid17r_ctrl
#SBATCH --partition=cpu-high
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%A_%a.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17r
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo";export PYTHONPATH="$repo/src";mkdir -p "$repo/.slurm/brainid_gate_v17r"
for ((index=SLURM_ARRAY_TASK_ID;index<52;index+=4)); do python -m eeg_cgdr.cli.brainid_gate_v17r --config configs/cgdr/brainid_gate_v17r.yaml --stage prepare-control --index "$index" --run-dir "results/cgdr/brainid_gate_v17r/server_runs/prepare-control/task_$index";done
