#!/bin/bash
#SBATCH --job-name=m13-w4
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.err
set -euo pipefail
m13=/home/infres/yinwang/denoiseNet_flagship_m0
v44=/home/infres/yinwang/denoiseNet_rgcc_eog_v44
# V44's eeg_scad first (model + helpers, read-only consumption); m13's eeg_chart second
export PYTHONPATH="$v44/src:$m13/src:$m13"
cd "$m13"
seeds=(20261201 20261202 20261203); cell=${SLURM_ARRAY_TASK_ID}; fold=$((cell/3)); seed_index=$((cell%3))
chains=8
if [ "$seed_index" -eq 0 ]; then chains=32; fi
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m13 w4 --fold "$fold" --seed "${seeds[$seed_index]}" --chains "$chains"
