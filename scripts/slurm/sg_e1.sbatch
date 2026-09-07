#!/bin/bash
#SBATCH --job-name=sg-e1
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e1_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e1_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/export_e1.py
