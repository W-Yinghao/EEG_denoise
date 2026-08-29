#!/bin/bash
#SBATCH --job-name=uqB-rebuild
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/uqB_rebuild_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/uqB_rebuild_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY scripts/iris/sealed_uq.py episodes
$PY scripts/iris/sealed_uq.py check
