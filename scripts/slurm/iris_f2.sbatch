#!/bin/bash
#SBATCH --job-name=iris-f2
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/f2_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/f2_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/iris/f2_fight.py
