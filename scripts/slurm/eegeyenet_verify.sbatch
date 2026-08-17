#!/bin/bash
#SBATCH --job-name=eegeyenet-verify
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/eegeyenet/verify_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/eegeyenet/verify_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
# Single job rather than an array: the QOS submit-per-user cap is saturated by the
# GPU queue, and 205 files is a ~15 min sequential read.
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/eegeyenet_verify.py \
  run --shard 0 --shards 1
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/eegeyenet_verify.py summarize
