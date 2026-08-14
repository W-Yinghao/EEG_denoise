#!/bin/bash
#SBATCH --job-name=v41r-natural
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/calib_eegdfus_v41r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/calib_eegdfus_v41r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_calib_eegdfus_v41r
export PYTHONPATH="$root/src:$root"
seeds=(20261110 20261111)
cell=${SLURM_ARRAY_TASK_ID:-0}; fold=$((cell / 2)); seed_index=$((cell % 2)); seed=${seeds[$seed_index]}
paired_run_id=${V41R_PAIRED_RUN_ID:?V41R_PAIRED_RUN_ID required}
run_id=${V41R_RUN_ID:-natural_job_${SLURM_ARRAY_JOB_ID:-manual}}
/home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v41r natural-freeze --fold "$fold" --seed "$seed" --paired-run-id "$paired_run_id" --run-id "$run_id"
/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python -m eeg_scad.cli.run_v41r natural-evaluate --fold "$fold" --seed "$seed" --run-id "$run_id"
