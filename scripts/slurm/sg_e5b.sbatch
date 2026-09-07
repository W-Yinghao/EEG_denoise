#!/bin/bash
#SBATCH --job-name=sg-e5b
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/e5b_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/e5b_%j.err
# E5b: held-out sub-04 paired exemplar with every method -> results/paper_final/paper_final_arrays/t6_heldout_methods_sub-04.npz
# Gates: G1 bit-identical contaminated/reference vs paper_final_arrays/t6_heldout_intervals_sub-04.npz,
#        G2 eog_drive allclose, G3 rebuilt matched arm within 5 % of the stored mean.
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /home/infres/yinwang/denoiseNet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/export_e5b.py
