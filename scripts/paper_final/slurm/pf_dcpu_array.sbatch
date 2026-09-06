#!/bin/bash
#SBATCH --job-name=pf-dcpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --array=0-4
#SBATCH --output=slurm_logs/paper_final/dcpu_%A_%a.out
#SBATCH --error=slurm_logs/paper_final/dcpu_%A_%a.err
set -euo pipefail
# Submit from the repo root (mkdir -p slurm_logs/paper_final first): the log
# paths above are relative to the submit directory. DENOISENET_ROOT and
# DENOISENET_PY override the repo root and the interpreter.
REPO_ROOT="${DENOISENET_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT/scripts/paper_final"
PY="${DENOISENET_PY:-$HOME/anaconda3/envs/icml/bin/python}"
$PY dwave.py cpu-arms --fold "$SLURM_ARRAY_TASK_ID"
