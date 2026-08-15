#!/bin/bash
#SBATCH --job-name=w4-e1r
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_wave4/scripts
/home/infres/yinwang/anaconda3/envs/icml/bin/python wave4_e1r.py
