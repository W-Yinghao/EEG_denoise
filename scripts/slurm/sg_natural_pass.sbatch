#!/bin/bash
#SBATCH --job-name=sg-natural-pass
#SBATCH --partition=V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/natural_pass_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/natural_pass_%j.err
# E2/E3/E6 job 1: rerun T5's exact 360 natural windows (seed 20261201, K=1) + ICA/ASR/SGEYESUB
# -> /projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass/<sub>|<ses>|<task>.npz
# Partition pinned to V100 on purpose: the stored V44-S1 stage-1 eval ran on V100
# (v44_stage1_eval.sbatch) and the bit-identity gate (1e-9 on the first cell) is only
# realistic on the same GPU class (Ampere+ cuDNN convs default to TF32).
set -euo pipefail
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /home/infres/yinwang/denoiseNet
nvidia-smi --query-gpu=name --format=csv,noheader || true
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/paper_final/sg_export/export_natural_pass.py
