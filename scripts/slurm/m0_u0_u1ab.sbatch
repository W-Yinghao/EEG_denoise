#!/bin/bash
#SBATCH --job-name=m0-u0-u1ab
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$py" -m eeg_chart.run_m0 u0a
"$py" -m eeg_chart.run_m0 u0b
"$py" -m eeg_chart.run_m0 u0c
"$py" -m eeg_chart.run_m0 u1a
"$py" -m eeg_chart.run_m0 u1b
