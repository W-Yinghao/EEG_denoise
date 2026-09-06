#!/bin/bash
#SBATCH --job-name=v43-stage15
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%A_%a.err
# A100/H100 partitions cap at 1 day; the instructed 36 h is unreachable there.
# V42R cells finished well under 2 h, so 24 h is a generous bound.
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
folds=(0 2); fold=${folds[$SLURM_ARRAY_TASK_ID]}
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v43 stage15 --fold "$fold" --seed 20261201 --updates 80000
