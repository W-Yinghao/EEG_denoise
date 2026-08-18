#!/bin/bash
#SBATCH --job-name=s356-fetch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/s356_fetch_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/s356_fetch_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
[ -f results/iris/s356/ext_manifest.json ] || $PY scripts/iris/s356_acquire.py manifest
$PY scripts/iris/s356_acquire.py fetch
