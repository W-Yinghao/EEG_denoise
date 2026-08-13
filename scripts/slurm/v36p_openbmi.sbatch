#!/usr/bin/env bash
#SBATCH --job-name=v36p-openbmi
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/fiber_openbmi_v36p/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/fiber_openbmi_v36p/%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/denoiseNet_fiber_openbmi_v36p
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p /home/infres/yinwang/slurm_logs/fiber_openbmi_v36p
seeds=(20260940 20260941)
fold=$((SLURM_ARRAY_TASK_ID / 2))
seed_index=$((SLURM_ARRAY_TASK_ID % 2))
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v36p run --fold "$fold" --seed "${seeds[$seed_index]}"
