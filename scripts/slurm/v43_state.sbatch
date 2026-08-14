#!/bin/bash
#SBATCH --job-name=v43-state
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v43 build-state
