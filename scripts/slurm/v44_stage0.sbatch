#!/bin/bash
#SBATCH --job-name=v44-stage0
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_eog_v44/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_eog_v44/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_eog_v44
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v44 stage0
