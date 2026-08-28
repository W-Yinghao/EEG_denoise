#!/bin/bash
#SBATCH --job-name=pf-t2f
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/t2f_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/t2f_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY t2_duration.py run --fold "$SLURM_ARRAY_TASK_ID"
