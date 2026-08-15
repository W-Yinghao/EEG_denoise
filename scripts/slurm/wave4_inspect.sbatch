#!/bin/bash
#SBATCH --job-name=w4-inspect
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_wave4
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/wave4_inspect_events.py
