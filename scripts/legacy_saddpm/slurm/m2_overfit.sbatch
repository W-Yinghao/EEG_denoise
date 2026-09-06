#!/bin/bash
#SBATCH --job-name=saddpm-m2
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/m2_overfit/slurm-%j.out

# M2 milestone gate: overfit one batch with the plain DDPM on a V100 GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/m2_overfit

PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"  # -u: unbuffered stdout for live logs
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}"
$PY scripts/check_env.py --require-cuda
$PY scripts/m2_overfit_batch.py "$@"
