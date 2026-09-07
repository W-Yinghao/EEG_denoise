#!/bin/bash
#SBATCH --job-name=sg-e5-fig
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e5_fig_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e5_fig_%j.err
# E5 figure: fig-sg-e5-exemplars (PDF+PNG under artifacts/figures/v3); submit with --dependency=afterok:<sg_e5 job>
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export MPLBACKEND=Agg
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/fig_sg_e5_exemplars.py
