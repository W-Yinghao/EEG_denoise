#!/bin/bash
#SBATCH --job-name=sg-e4
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e4_export_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e4_export_%j.err
# E4: ERP grand averages + P300 peak topographies -> results/paper_final/paper_final_arrays/erp_topographies.npz
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/export_e4.py
