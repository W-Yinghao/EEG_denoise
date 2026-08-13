#!/usr/bin/env bash
#SBATCH --job-name=v40r-sc-eegdfus
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/official_support_diffusion_v40r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/official_support_diffusion_v40r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_official_support_diffusion_v40r
mkdir -p /home/infres/yinwang/slurm_logs/official_support_diffusion_v40r
cd "$root"
export PYTHONPATH="$root/src:$root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
seeds=(20261010 20261011)
fold=$((SLURM_ARRAY_TASK_ID/2))
seed_index=$((SLURM_ARRAY_TASK_ID%2))
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v40r run --fold "$fold" --seed "${seeds[$seed_index]}"
