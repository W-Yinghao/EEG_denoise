#!/bin/bash
#SBATCH --job-name=pf-t3
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/t3_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/t3_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet/scripts/paper_final
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY t3_ablation.py run
$PY t3_ablation.py aggregate
