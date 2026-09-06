#!/bin/bash
#SBATCH --job-name=pf-dwave-h
#SBATCH --partition=A100,H100,L40S,A40,V100
#SBATCH --exclude=node54
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:59:59
#SBATCH --output=slurm_logs/paper_final/dwave_h_%j.out
#SBATCH --error=slurm_logs/paper_final/dwave_h_%j.err
set -euo pipefail
# Submit from the repo root (mkdir -p slurm_logs/paper_final first): the log
# paths above are relative to the submit directory. DENOISENET_ROOT and
# DENOISENET_PY override the repo root and the interpreter.
REPO_ROOT="${DENOISENET_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT/scripts/paper_final"
PY="${DENOISENET_PY:-$HOME/anaconda3/envs/icml/bin/python}"
$PY dwave.py d1 --heldout
$PY dwave.py d2 --heldout
