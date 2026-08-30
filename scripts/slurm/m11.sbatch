#!/bin/bash
#SBATCH --job-name=saddpm-m11
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/m11/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/m11
PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}"
$PY scripts/check_env.py --require-cuda
$PY scripts/m11_mc_baselines.py "$@"
