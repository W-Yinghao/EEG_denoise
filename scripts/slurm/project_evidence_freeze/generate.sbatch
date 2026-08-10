#!/usr/bin/env bash
#SBATCH --job-name=evidence_freeze
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_project_evidence_freeze/.slurm/project_evidence_freeze/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_project_evidence_freeze/.slurm/project_evidence_freeze/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_project_evidence_freeze
mkdir -p "$repo/.slurm/project_evidence_freeze"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo"
PYTHONPATH=src python -m eeg_cgdr.cli.project_evidence_freeze --repo "$repo"
