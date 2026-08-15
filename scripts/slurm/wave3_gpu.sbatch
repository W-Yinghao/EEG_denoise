#!/bin/bash
#SBATCH --job-name=w3-gpu
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave3/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave3/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave3
export PYTHONPATH="/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src:$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
unit=${W3_UNIT:?W3_UNIT required}
"$py" -m eeg_chart.run_wave3_gpu "$unit"
