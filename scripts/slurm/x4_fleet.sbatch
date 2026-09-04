#!/bin/bash
#SBATCH --job-name=w6-e4
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --array=0-4
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave6/e4_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave6/e4_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
for SEED in 20261201 20261202 20261203; do
  $PY x4_recalibration.py run --fold "$SLURM_ARRAY_TASK_ID" --seed "$SEED"
done
