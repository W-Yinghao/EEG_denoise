#!/bin/bash
#SBATCH --job-name=dsddpm-ica
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/dsddpm_ica_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/dsddpm_ica_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/dsddpm_repro.py ica
