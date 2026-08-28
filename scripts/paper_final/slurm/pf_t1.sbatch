#!/bin/bash
#SBATCH --job-name=pf-t1
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/t1_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/t1_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY t1_heldout_uq.py run
$PY t1_heldout_uq.py aggregate
