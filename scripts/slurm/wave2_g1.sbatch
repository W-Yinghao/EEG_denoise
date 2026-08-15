#!/bin/bash
#SBATCH --job-name=w2-g1
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave2/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave2/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_wave2
export PYTHONPATH="/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src:$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_wave2_gpu g1 --fold "${SLURM_ARRAY_TASK_ID}"
