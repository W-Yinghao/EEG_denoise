#!/usr/bin/env bash
#SBATCH --job-name=brainid17r_test
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17r
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo";export PYTHONPATH="$repo/src";mkdir -p "$repo/.slurm/brainid_gate_v17r"
pytest -q tests/unit/test_brainid_gate_v17r.py
python -c 'import eeg_cgdr.experiments.brainid_gate_v17r; import eeg_cgdr.cli.brainid_gate_v17r'
