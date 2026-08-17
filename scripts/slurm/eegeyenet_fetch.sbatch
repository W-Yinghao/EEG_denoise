#!/bin/bash
#SBATCH --job-name=eegeyenet-fetch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/infres/yinwang/slurm_logs/eegeyenet/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/eegeyenet/%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/eegeyenet_download.py \
  run --shard "$SLURM_ARRAY_TASK_ID" --shards 6
