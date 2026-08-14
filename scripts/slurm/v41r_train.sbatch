#!/bin/bash
#SBATCH --job-name=v41r-calib-eegdfus
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/calib_eegdfus_v41r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/calib_eegdfus_v41r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_calib_eegdfus_v41r
mkdir -p /home/infres/yinwang/slurm_logs/calib_eegdfus_v41r
export PYTHONPATH="$root/src:$root"
seeds=(20261110 20261111)
cell=${SLURM_ARRAY_TASK_ID:-0}
fold=$((cell / 2))
seed_index=$((cell % 2))
updates=${V41R_UPDATES:-12000}
run_id=${V41R_RUN_ID:-job_${SLURM_ARRAY_JOB_ID:-manual}}
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v41r run \
  --fold "$fold" --seed "${seeds[$seed_index]}" --run-id "$run_id" --updates "$updates"
