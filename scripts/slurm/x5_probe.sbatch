#!/bin/bash
#SBATCH --job-name=w6-e5probe
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave6/e5probe_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave6/e5probe_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
export PYTHONPATH=/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src
/home/infres/yinwang/anaconda3/envs/icml/bin/python x5_calibration_content.py probe --fold 0 --seed 20261201
