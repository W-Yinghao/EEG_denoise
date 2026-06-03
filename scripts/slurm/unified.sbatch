#!/bin/bash
#SBATCH --job-name=saddpm-unif
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/unified/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/unified
PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"
$PY scripts/check_env.py --require-cuda
$PY scripts/unified_compare.py "$@"
