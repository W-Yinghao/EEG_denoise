#!/bin/bash
#SBATCH --job-name=v42r-natural
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_calib_saddpm_cond_v42r
export PYTHONPATH="$root/src:$root"
seeds=(20261201 20261202)
cell=${SLURM_ARRAY_TASK_ID}
fold=$((cell/2))
seed_index=$((cell%2))
paired_run_id=${V42R_PAIRED_RUN_ID:?V42R_PAIRED_RUN_ID is required}
run_id=${V42R_NATURAL_RUN_ID:?V42R_NATURAL_RUN_ID is required}
python_bin=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$python_bin" -m eeg_scad.cli.run_v42r paired-channel --fold "$fold" --seed "${seeds[$seed_index]}" --paired-run-id "$paired_run_id"
"$python_bin" -m eeg_scad.cli.run_v42r natural-freeze --fold "$fold" --seed "${seeds[$seed_index]}" --paired-run-id "$paired_run_id" --run-id "$run_id"
"$python_bin" -m eeg_scad.cli.run_v42r natural-evaluate --fold "$fold" --seed "${seeds[$seed_index]}" --run-id "$run_id"
