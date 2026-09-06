#!/bin/bash
#SBATCH --job-name=pf-t2
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=slurm_logs/paper_final/t2_%j.out
#SBATCH --error=slurm_logs/paper_final/t2_%j.err
set -euo pipefail
# Submit from the repo root (mkdir -p slurm_logs/paper_final first): the log
# paths above are relative to the submit directory. DENOISENET_ROOT and
# DENOISENET_PY override the repo root and the interpreter.
REPO_ROOT="${DENOISENET_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT/scripts/paper_final"
PY="${DENOISENET_PY:-$HOME/anaconda3/envs/icml/bin/python}"
$PY t2_duration.py run
$PY t2_duration.py aggregate
