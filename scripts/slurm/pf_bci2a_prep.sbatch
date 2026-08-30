#!/bin/bash
#SBATCH --job-name=2a-prep
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/bci2a_prep_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/bci2a_prep_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
$PY bci2a.py prep
$PY bci2a.py operators
$PY bci2a.py episodes
