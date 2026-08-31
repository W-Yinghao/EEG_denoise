#!/bin/bash
#SBATCH --job-name=pf-table1
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --array=0-1
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/table1_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/table1_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0) $PY scripts/paper_final/dsddpm_table1.py raw ;;
  1) $PY scripts/paper_final/dsddpm_table1.py ica ;;
esac
