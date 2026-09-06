#!/bin/bash
#SBATCH --job-name=v42r-native-x0
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_calib_saddpm_cond_v42r
mkdir -p /home/infres/yinwang/slurm_logs/calib_saddpm_cond_v42r
export PYTHONPATH="$root/src:$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v42r native
