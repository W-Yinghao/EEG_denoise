#!/bin/bash
#SBATCH --job-name=v42r-full
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_calib_saddpm_cond_v42r
mkdir -p /home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r
export PYTHONPATH="$root/src:$root"
seeds=(20261201 20261202); cell=${SLURM_ARRAY_TASK_ID}; fold=$((cell/2)); seed_index=$((cell%2)); run_id=${V42R_RUN_ID:-job_${SLURM_ARRAY_JOB_ID}}
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v42r run --fold "$fold" --seed "${seeds[$seed_index]}" --updates 80000 --run-id "$run_id"
