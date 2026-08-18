#!/bin/bash
#SBATCH --job-name=iris-f4agg
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/f4agg_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/f4agg_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/iris/f4_uq.py aggregate
