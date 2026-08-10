#!/usr/bin/env bash
#SBATCH --job-name=evidence_clean
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_project_evidence_freeze/.slurm/project_evidence_freeze/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_project_evidence_freeze/.slurm/project_evidence_freeze/%j.err
set -euo pipefail
source_repo=/home/infres/yinwang/denoiseNet_project_evidence_freeze
test_root=$(mktemp -d /tmp/evidence-freeze-clean.XXXXXX)
trap 'rm -rf "$test_root"' EXIT
git clone --quiet --shared "$source_repo" "$test_root/repo"
cd "$test_root/repo"
git checkout --quiet "$EXPECTED_COMMIT"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
PYTHONPATH=src python -c 'import eeg_cgdr.experiments.project_evidence_freeze'
PYTHONPATH=src pytest -q tests/unit/test_project_evidence_freeze.py tests/unit/test_raw_support_closure.py
test "$(git status --porcelain)" = ""
