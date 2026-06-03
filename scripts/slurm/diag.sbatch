#!/bin/bash
#SBATCH --job-name=saddpm-diag
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/diag/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/diag
PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"
$PY scripts/diag_eegnet.py
