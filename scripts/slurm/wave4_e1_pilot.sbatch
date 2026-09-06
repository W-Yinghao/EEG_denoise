#!/bin/bash
#SBATCH --job-name=w4-e1-pilot
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_wave4
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$py" scripts/wave4_e1_align.py diagnose --limit 4
"$py" scripts/wave4_e1_align.py pilot
