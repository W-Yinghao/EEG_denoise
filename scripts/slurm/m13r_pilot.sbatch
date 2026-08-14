#!/bin/bash
#SBATCH --job-name=m13r-pilot
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
panels=(mobilebci klados bci2b)
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_chart.run_m13 m13r-pilot --panel "${panels[$SLURM_ARRAY_TASK_ID]}"
