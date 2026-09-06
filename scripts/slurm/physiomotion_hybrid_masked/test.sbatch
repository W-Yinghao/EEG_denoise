#!/usr/bin/env bash
#SBATCH --job-name=phyhyb_test
#SBATCH --partition=CPU
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked/results/cgdr/physiomotion_hybrid_masked/slurm/%j_test.out
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
python -m pytest -q tests/unit/test_physiomotion_hybrid_masked.py
python - <<'PY'
from eeg_cgdr.experiments.physiomotion_hybrid_masked import STAGES
from eeg_cgdr.models.hybrid_masked_diffusion import HybridMaskedConfig
assert {'freeze','materialize','technical','train','infer','evaluate','aggregate'} <= set(STAGES)
HybridMaskedConfig()
PY
