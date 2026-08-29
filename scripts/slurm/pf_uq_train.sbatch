#!/bin/bash
#SBATCH --job-name=uqB-train
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/uqB_train_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/uqB_train_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/sealed_uq.py train
$PY scripts/iris/sealed_uq.py freeze-temp
