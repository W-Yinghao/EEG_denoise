#!/usr/bin/env bash
#SBATCH --job-name=trait18_test
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo"
export PYTHONPATH="$repo/src"
mkdir -p "$repo/.slurm/physiotrait_actionability_v18"
pytest -q tests/unit/test_physiotrait_actionability_v18.py
python -c 'import eeg_cgdr.experiments.physiotrait_actionability_v18; import eeg_cgdr.cli.physiotrait_actionability_v18'
