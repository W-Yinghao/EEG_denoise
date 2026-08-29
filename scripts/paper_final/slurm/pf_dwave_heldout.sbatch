#!/bin/bash
#SBATCH --job-name=pf-dwave-h
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/dwave_h_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/dwave_h_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY dwave.py d1 --heldout
$PY dwave.py d2 --heldout
