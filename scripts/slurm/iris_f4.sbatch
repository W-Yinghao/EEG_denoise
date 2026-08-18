#!/bin/bash
#SBATCH --job-name=iris-f4
#SBATCH --partition=A100,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/f4_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/f4_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/f4_uq.py run
$PY scripts/iris/f4_uq.py aggregate
