#!/usr/bin/env bash
#SBATCH --job-name=v35p-channel
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/home/infres/yinwang/slurm_logs/fiber_channel_v35p/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/fiber_channel_v35p/%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/denoiseNet_fiber_channel_v35p
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p /home/infres/yinwang/slurm_logs/fiber_channel_v35p
seeds=(20260920 20260921)
fold=$((SLURM_ARRAY_TASK_ID / 2))
seed_index=$((SLURM_ARRAY_TASK_ID % 2))
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v35p run --fold "$fold" --seed "${seeds[$seed_index]}"
