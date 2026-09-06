#!/bin/bash
#SBATCH --job-name=pf-cpu-analysis
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --output=slurm_logs/paper_final/cpu_analysis_%j.out
#SBATCH --error=slurm_logs/paper_final/cpu_analysis_%j.err
set -euo pipefail
# Submit from the repo root (mkdir -p slurm_logs/paper_final first): the log
# paths above are relative to the submit directory. DENOISENET_ROOT and
# DENOISENET_PY override the repo root and the interpreter.
REPO_ROOT="${DENOISENET_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT/scripts/paper_final"
PY="${DENOISENET_PY:-$HOME/anaconda3/envs/icml/bin/python}"
$PY t5_natural_plane.py
$PY t4_staleness.py cpu
$PY t6_extras.py
