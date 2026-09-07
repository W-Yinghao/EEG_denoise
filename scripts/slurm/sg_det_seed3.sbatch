#!/bin/bash
#SBATCH --job-name=sg-det-s3
#SBATCH --partition=A100,H100
#SBATCH --exclude=node54,nodeaudible01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:59:59
#SBATCH --array=0-4
#SBATCH --output=/home/infres/yinwang/slurm_logs/sg/det_s3_%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/sg/det_s3_%A_%a.err
# Item B: deterministic twin for seed 20261203, exactly the stage-1 det recipe (80000 updates),
# run from the self-contained consolidated tree (src/eeg_scad byte-identical to codex/rgcc-eog-v44).
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_consolidated
export PYTHONPATH=/home/infres/yinwang/denoiseNet_consolidated/src
echo "node=$(hostname)"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v44 stage1-det --fold "$SLURM_ARRAY_TASK_ID" --seed 20261203 --updates 80000
