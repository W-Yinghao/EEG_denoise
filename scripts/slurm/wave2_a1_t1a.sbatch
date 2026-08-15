#!/bin/bash
#SBATCH --job-name=w2-a1-t1a
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave2/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave2/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave2
export PYTHONPATH="/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src:$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$py" -m eeg_chart.run_wave2_gpu a1
"$py" -m eeg_chart.run_wave2_gpu t1a
