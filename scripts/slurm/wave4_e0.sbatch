#!/bin/bash
#SBATCH --job-name=w4-e0-manifest
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave4
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/wave4_e0_manifest.py
