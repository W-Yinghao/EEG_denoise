#!/bin/bash
#SBATCH --job-name=m0-decision
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m0 decision
