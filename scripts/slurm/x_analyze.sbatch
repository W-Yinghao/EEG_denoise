#!/bin/bash
#SBATCH --job-name=w6-an
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave6/an_%x_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave6/an_%x_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
case "$WHICH" in
  e3) $PY x3_event_selectivity.py ;;
  *)  $PY x_analyze.py "$WHICH" ;;
esac
