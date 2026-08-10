#!/usr/bin/env bash
#SBATCH --job-name=trait18_prep
#SBATCH --partition=cpu-high
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
for offset in 0 1 2; do
    index=$((SLURM_ARRAY_TASK_ID * 3 + offset))
    python -m eeg_cgdr.cli.physiotrait_actionability_v18 --config configs/cgdr/physiotrait_actionability_v18.yaml --stage prepare --index "$index" --run-dir "results/cgdr/physiotrait_actionability_v18/server_runs/prepare/task_$index"
done
