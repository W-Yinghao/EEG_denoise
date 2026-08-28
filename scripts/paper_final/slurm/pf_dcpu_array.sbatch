#!/bin/bash
#SBATCH --job-name=pf-dcpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --array=0-4
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/dcpu_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/dcpu_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY dwave.py cpu-arms --fold "$SLURM_ARRAY_TASK_ID"
