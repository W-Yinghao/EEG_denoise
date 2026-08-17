#!/bin/bash
#SBATCH --job-name=iris-w
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/w_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/w_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/iris/w_stage.py
