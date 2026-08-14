#!/bin/bash
#SBATCH --job-name=m13-harvest
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m13 w2-harvest
