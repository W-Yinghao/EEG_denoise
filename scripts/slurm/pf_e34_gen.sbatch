#!/bin/bash
#SBATCH --job-name=pf-e4gen
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/e4gen_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/e4gen_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/e34_ssed_ours.py gen
