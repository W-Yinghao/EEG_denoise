#!/bin/bash
#SBATCH --job-name=iris-sealed-fetch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/sealed_fetch_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/sealed_fetch_%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet
# Downloads the frozen sealed block into quarantine, header-verifies (acquisition
# integrity only), then chmod 000 the tree. No analysis touches these bytes.
/home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/iris/freeze_sealed.py fetch
