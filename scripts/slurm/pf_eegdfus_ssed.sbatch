#!/bin/bash
#SBATCH --job-name=dfus-ssed
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/dfus_ssed_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/dfus_ssed_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
$PY eegdfus_ssed.py train
$PY eegdfus_ssed.py eval
