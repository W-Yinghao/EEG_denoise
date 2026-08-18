#!/bin/bash
#SBATCH --job-name=s356-prep
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/s356_prep_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/s356_prep_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/s356_probe.py prep
$PY scripts/iris/s356_probe.py episodes
