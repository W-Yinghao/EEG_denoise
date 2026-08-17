#!/bin/bash
#SBATCH --job-name=iris-k
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/k_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/k_%j.err
set -uo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
# Sequential single job: QOS submit headroom is held by the operator's GPU fleet.
# set -e intentionally absent: one instrument's failure must not silence the others.
echo "=== K1 abstention taxonomy ==="; $PY scripts/iris/k1_abstention_taxonomy.py
echo "=== K2 sigma_drift recheck ===";  $PY scripts/iris/k2_sigma_drift.py
echo "=== K3 EEGEyeNet instrument ==="; $PY scripts/iris/k3_instrument.py
