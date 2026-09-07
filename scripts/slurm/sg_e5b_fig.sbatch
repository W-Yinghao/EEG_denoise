#!/bin/bash
#SBATCH --job-name=sg-e5b-fig
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e5b_fig_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e5b_fig_%j.err
# E5b figures: fig-sg-e5b-methods + fig-sg-e5b-methods-ref (PDF+PNG under artifacts/figures/v3),
# windows 1,4,6 0-based; submit with --dependency=afterok:<sg_e5b job>
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export MPLBACKEND=Agg
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/fig_sg_e5b_methods.py --windows 1,4,6
