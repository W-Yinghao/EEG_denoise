#!/bin/bash
#SBATCH --job-name=v43-s2b
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
folds=(1 3 4); fold=${folds[$SLURM_ARRAY_TASK_ID]}
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v43 stage15 --fold "$fold" --seed 20261201 --updates 80000
