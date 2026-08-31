#!/bin/bash
#SBATCH --job-name=pf-t1norm
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --array=0-5
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/t1norm_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/t1norm_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
ARMS=(raw ica dsddpm raw ica dsddpm)
NORMS=(l2 l2 l2 zscore zscore zscore)
$PY scripts/paper_final/dsddpm_table1.py "${ARMS[$SLURM_ARRAY_TASK_ID]}" --recipe adam --norm "${NORMS[$SLURM_ARRAY_TASK_ID]}"
