#!/bin/bash
#SBATCH --job-name=v43-s3-agg
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/rgcc_v43/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/rgcc_v43/%j.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_rgcc_v43
export PYTHONPATH="$root/src:$root"
cd "$root"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v43_s3 aggregate
