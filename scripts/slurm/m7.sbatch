#!/bin/bash
#SBATCH --job-name=saddpm-m7
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/runs/m7/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/m7
PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}"
$PY scripts/check_env.py --require-cuda
$PY scripts/run_pairwise_matrix.py "$@"
