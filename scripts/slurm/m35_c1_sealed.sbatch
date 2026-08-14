#!/bin/bash
#SBATCH --job-name=m35-c1-sealed
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%j.err
# SINGLE-PASS SEALED CHAIN — do not resubmit on failure without operator review.
set -euo pipefail
m13=/home/infres/yinwang/denoiseNet_flagship_m0
v44=/home/infres/yinwang/denoiseNet_rgcc_eog_v44
export PYTHONPATH="$v44/src:$m13/src:$m13"
cd "$m13"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m35_sealed
