#!/bin/bash
#SBATCH --job-name=s356-train
#SBATCH --partition=A100,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/s356_train_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/s356_train_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
for n in 30 60 120 200 -1; do
  for arm in COND BLIND; do
    $PY scripts/iris/s356_probe.py train --n $n --arm $arm
  done
done
$PY scripts/iris/s356_probe.py evaluate
$PY scripts/iris/s356_probe.py aggregate
