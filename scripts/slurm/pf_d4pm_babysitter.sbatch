#!/bin/bash
#SBATCH --job-name=d4pm-sitter
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:59:59
#SBATCH --output=/home/infres/yinwang/slurm_logs/paper_final/d4pm_sitter_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/paper_final/d4pm_sitter_%j.err
set -uo pipefail
exec /home/infres/yinwang/denoiseNet/scripts/slurm/d4pm_babysitter.sh
