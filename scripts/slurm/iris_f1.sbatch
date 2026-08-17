#!/bin/bash
#SBATCH --job-name=iris-f1
#SBATCH --partition=A100,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/f1_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/f1_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/f1_diffusion_fallback.py run
$PY scripts/iris/f1_diffusion_fallback.py aggregate
