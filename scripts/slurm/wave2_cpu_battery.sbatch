#!/bin/bash
#SBATCH --job-name=w2-cpu-battery
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave2/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave2/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave2
export PYTHONPATH="/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src:$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$py" -m eeg_chart.run_wave2 shared-layer
"$py" -m eeg_chart.run_wave2 p2
"$py" -m eeg_chart.run_wave2 moka
"$py" -m eeg_chart.run_wave2 opera-a0
"$py" -m eeg_chart.run_wave2 gibbs-g0
"$py" -m eeg_chart.run_wave2 thresh-t0
