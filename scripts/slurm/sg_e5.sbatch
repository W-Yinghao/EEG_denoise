#!/bin/bash
#SBATCH --job-name=sg-e5
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e5_export_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e5_export_%j.err
# E5: one natural ERP tile per movement condition for sub-02 -> results/paper_final/paper_final_arrays/natural_exemplars.npz
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/export_e5.py
