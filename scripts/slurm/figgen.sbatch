#!/bin/bash
#SBATCH --job-name=saddpm-fig
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=artifacts/runs/figgen/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p artifacts/runs/figgen
PY="/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -u"
$PY scripts/figgen_eval.py "$@"
