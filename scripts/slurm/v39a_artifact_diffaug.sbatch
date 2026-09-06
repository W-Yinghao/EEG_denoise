#!/usr/bin/env bash
#SBATCH --job-name=v39a-diffaug
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/artifact_diffaug_v39a/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/artifact_diffaug_v39a/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_artifact_diffaug_v39a
mkdir -p /home/infres/yinwang/slurm_logs/artifact_diffaug_v39a
cd "$root";export PYTHONPATH="$root/src:$root";export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
seeds=(20260950 20260951);fold=$((SLURM_ARRAY_TASK_ID/2));si=$((SLURM_ARRAY_TASK_ID%2))
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v39a run --fold "$fold" --seed "${seeds[$si]}"
