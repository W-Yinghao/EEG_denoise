#!/bin/bash
#SBATCH --job-name=eegeyenet-man
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/eegeyenet/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/eegeyenet/%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/eegeyenet_download.py manifest
