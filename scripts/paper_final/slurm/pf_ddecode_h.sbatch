#!/bin/bash
#SBATCH --job-name=pf-ddec-h
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/ddecode_h_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/ddecode_h_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY dwave.py decode --heldout
