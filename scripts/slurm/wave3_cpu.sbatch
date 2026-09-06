#!/bin/bash
#SBATCH --job-name=w3-cpu
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave3/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave3/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave3
export PYTHONPATH="/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src:$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== W0 ==="
"$py" -m eeg_chart.run_wave3 b0
"$py" -m eeg_chart.run_wave3 t0
"$py" -m eeg_chart.run_wave3 once0
"$py" -m eeg_chart.run_wave3 packa-fp
echo "=== W1-2 ==="
"$py" -m eeg_chart.run_wave3 b15
"$py" -m eeg_chart.run_wave3 b1
"$py" -m eeg_chart.run_wave3 t7
"$py" -m eeg_chart.run_wave3 t1
"$py" -m eeg_chart.run_wave3 t6
"$py" -m eeg_chart.run_wave3 t4
"$py" -m eeg_chart.run_wave3 once12
