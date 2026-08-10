#!/usr/bin/env bash
#SBATCH --job-name=shu_j0_data
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=reports/slurm/shu_task_phenotype/%j_j0_data.out

set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd /home/infres/yinwang/denoiseNet_shu_task_phenotype
python scripts/slurm/shu_task_phenotype/audit_datalake.py \
  --lmdb /projects/EEG-foundation-model/tdoan-24/SHUMI_256hz \
  --output results/cgdr/shu_task_phenotype_diffusion/j0/datalake_inventory.json
