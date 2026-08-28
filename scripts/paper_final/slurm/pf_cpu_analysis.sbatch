#!/bin/bash
#SBATCH --job-name=pf-cpu-analysis
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/cpu_analysis_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/cpu_analysis_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY t5_natural_plane.py
$PY t4_staleness.py cpu
$PY t6_extras.py
