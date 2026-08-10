#!/usr/bin/env bash
#SBATCH --job-name=brainid_tests
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd "$repo"; export PYTHONPATH="$repo/src"
pytest -q tests/unit/test_brainid_gate_v17.py
python -c 'import eeg_cgdr.experiments.brainid_gate_v17; import eeg_cgdr.models.brainid_verifier_b'
