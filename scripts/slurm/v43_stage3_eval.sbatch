#!/bin/bash
#SBATCH --job-name=v43-s3-eval
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
seeds=(20261201 20261202 20261203); cell=${SLURM_ARRAY_TASK_ID}; fold=$((cell/3)); seed=${seeds[$((cell%3))]}
"$py" -m eeg_scad.cli.run_v43_s3 stage3-eval --fold "$fold" --seed "$seed"
"$py" -m eeg_scad.cli.run_v43_s3 stage3-natural-freeze --fold "$fold" --seed "$seed"
"$py" -m eeg_scad.cli.run_v43_s3 stage3-natural-evaluate --fold "$fold" --seed "$seed"
