#!/bin/bash
#SBATCH --job-name=m13r-w3
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m13 m13r-w3
