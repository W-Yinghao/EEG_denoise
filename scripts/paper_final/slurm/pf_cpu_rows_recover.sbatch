#!/bin/bash
#SBATCH --job-name=pf-cpu-rec
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/cpu_rec_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/cpu_rec_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY cpu_rows.py asr
$PY cpu_rows.py ica
$PY cpu_rows.py aggregate
