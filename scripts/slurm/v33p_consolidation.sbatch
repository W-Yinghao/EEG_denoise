#!/usr/bin/env bash
#SBATCH --job-name=v33p-refit
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/home/infres/yinwang/slurm_logs/sandiff_v33p/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sandiff_v33p/%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/denoiseNet_sandiff_consolidation_v33p
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p /home/infres/yinwang/slurm_logs/sandiff_v33p
seeds=(20260920 20260921)
fold=$((SLURM_ARRAY_TASK_ID / 2))
seed_index=$((SLURM_ARRAY_TASK_ID % 2))
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v33p run --fold "$fold" --seed "${seeds[$seed_index]}"
