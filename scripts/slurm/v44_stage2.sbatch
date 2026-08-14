#!/bin/bash
#SBATCH --job-name=v44-s2
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_eog_v44/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_eog_v44/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_eog_v44
export PYTHONPATH="$root/src:$root"
cd "$root"
seeds=(20261201 20261202 20261203); cell=${SLURM_ARRAY_TASK_ID}; fold=$((cell/3)); seed_index=$((cell%3))
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v44_s2 stage2 --fold "$fold" --seed "${seeds[$seed_index]}"
