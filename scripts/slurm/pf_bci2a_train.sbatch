#!/bin/bash
#SBATCH --job-name=2a-train
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --array=0-2
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/bci2a_train_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/bci2a_train_%A_%a.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
/home/infres/yinwang/anaconda3/envs/icml/bin/python bci2a.py train --fold "$SLURM_ARRAY_TASK_ID"
