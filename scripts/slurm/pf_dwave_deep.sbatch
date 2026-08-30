#!/bin/bash
#SBATCH --job-name=dwave-deep
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --array=0-3
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/dwave_deep_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/dwave_deep_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0) $PY dwave_deep.py SSVEP ;;
  1) $PY dwave_deep.py ERP ;;
  2) $PY dwave_deep.py SSVEP --heldout ;;
  3) $PY dwave_deep.py ERP --heldout ;;
esac
