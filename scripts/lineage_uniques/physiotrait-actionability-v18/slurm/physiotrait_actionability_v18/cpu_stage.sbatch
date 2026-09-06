#!/usr/bin/env bash
#SBATCH --job-name=trait18_cpu
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%A_%a.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%A_%a.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo"
export PYTHONPATH="$repo/src"
mkdir -p "$repo/.slurm/physiotrait_actionability_v18"
stage="${STAGE:?}"
task="${SLURM_ARRAY_TASK_ID:-0}"
run="$repo/results/cgdr/physiotrait_actionability_v18/server_runs/$stage/task_$task"
python -m eeg_cgdr.cli.physiotrait_actionability_v18 --config configs/cgdr/physiotrait_actionability_v18.yaml --stage "$stage" --index "$task" --run-dir "$run"
