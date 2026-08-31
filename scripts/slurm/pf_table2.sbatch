#!/bin/bash
#SBATCH --job-name=pf-table2
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/table2_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/table2_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/dsddpm_table2.py
