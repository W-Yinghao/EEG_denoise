#!/bin/bash
#SBATCH --job-name=w4-e1
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/wave4/%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/wave4/%j.err
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_wave4
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mode=${W4_MODE:?W4_MODE required}
if [ "$mode" = "diagnose" ]; then exec "$py" scripts/wave4_e1_align.py diagnose --limit 4; fi
if [ "$mode" = "pilot" ]; then exec "$py" scripts/wave4_e1_align.py pilot; fi
exec "$py" scripts/wave4_e1_align.py all
