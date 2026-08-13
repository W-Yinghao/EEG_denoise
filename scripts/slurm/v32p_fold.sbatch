#!/usr/bin/env bash
#SBATCH --job-name=v32p-fold
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/infres/yinwang/slurm_logs/sandiff_v32p/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sandiff_v32p/%A_%a.err

set -euo pipefail
cd /home/infres/yinwang/denoiseNet_sandiff_v32p
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p /home/infres/yinwang/slurm_logs/sandiff_v32p
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v32p fold --fold "${SLURM_ARRAY_TASK_ID}"
