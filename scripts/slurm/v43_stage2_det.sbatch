#!/bin/bash
#SBATCH --job-name=v43-s2-det
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
seeds=(20261201 20261202); cell=${SLURM_ARRAY_TASK_ID}; fold=$((cell/2)); seed_index=$((cell%2))
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v43 stage2-det --fold "$fold" --seed "${seeds[$seed_index]}" --updates 80000
