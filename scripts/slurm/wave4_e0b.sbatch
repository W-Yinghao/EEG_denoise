#!/bin/bash
#SBATCH --job-name=w4-download
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_wave4
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/wave4_e0b_download.py download --shard "${SLURM_ARRAY_TASK_ID}" --shards 8
