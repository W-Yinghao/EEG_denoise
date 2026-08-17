#!/bin/bash
#SBATCH --job-name=iris-p1r-w
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/p1r_w_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/p1r_w_%j.err
set -uo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
echo "=== P1R fallback repair ===";        $PY scripts/iris/p1r_fallback.py
echo "=== W/T repaired encoding + T2 ==="; $PY scripts/iris/w_stage.py --repaired
