#!/bin/bash
#SBATCH --job-name=s356-eval2
#SBATCH --partition=A100,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/s356_eval2_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/s356_eval2_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/s356_probe.py evaluate2
$PY scripts/iris/s356_probe.py aggregate2
