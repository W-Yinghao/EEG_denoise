#!/bin/bash
#SBATCH --job-name=sg-natural-post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/natural_post_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/natural_post_%j.err
# E2/E3/E6 job 2 (submit with --dependency=afterok:<sg_natural_pass job>):
#   natural_residual_corr.npz, natural_psd_ratio.npz, natural_band_power.npz
#   -> results/paper_final/paper_final_arrays/, then the three figures
#   (fig-sg-e2-residual-corr[-high], fig-sg-e3-psd-ratio, fig-sg-e6-band-power) under artifacts/figures/v3
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/paper_final/sg_export/export_natural_post.py
$PY scripts/paper_final/sg_export/fig_sg_e2_residual_corr.py
$PY scripts/paper_final/sg_export/fig_sg_e3_psd_ratio.py
$PY scripts/paper_final/sg_export/fig_sg_e6_band_power.py
