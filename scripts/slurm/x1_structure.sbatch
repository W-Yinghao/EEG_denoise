#!/bin/bash
#SBATCH --job-name=w6-e1
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave6/e1_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave6/e1_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
/home/infres/yinwang/anaconda3/envs/icml/bin/python x1_operator_structure.py
