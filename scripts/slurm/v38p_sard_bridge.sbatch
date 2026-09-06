#!/usr/bin/env bash
#SBATCH --job-name=v38p-sard
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/sard_bridge_v38p/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sard_bridge_v38p/%A_%a.err

set -euo pipefail
readonly code_root=/home/infres/yinwang/denoiseNet_sard_bridge_v38p
readonly python_bin=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/slurm_logs/sard_bridge_v38p
cd "$code_root"
export PYTHONPATH="$code_root/src:$code_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
seeds=(20260940 20260941)
fold=$((SLURM_ARRAY_TASK_ID / 2))
seed_index=$((SLURM_ARRAY_TASK_ID % 2))
exec "$python_bin" -m eeg_scad.cli.run_v38p run-fold --fold "$fold" --seed "${seeds[$seed_index]}" --device cuda
