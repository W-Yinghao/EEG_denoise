#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_raw_support_closure
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
stage=$1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
case "$stage" in
 aggregate) "$PY" -m eeg_cgdr.cli.raw_support_closure --source /home/infres/yinwang/denoiseNet_raw_support_clean_diffusion/results/cgdr/raw_support_clean_diffusion --output results/cgdr/raw_support_clean_diffusion_closure --report reports/raw_support_clean_diffusion_closure.md ;;
 tests) "$PY" -m pytest -q tests/unit/test_raw_support_closure.py ;;
 clean-import) tmp=${SLURM_TMPDIR:-/tmp}/closure_${SLURM_JOB_ID}; mkdir -p "$tmp"; git archive HEAD | tar -x -C "$tmp"; PYTHONPATH="$tmp/src:$tmp" "$PY" -c 'import eeg_cgdr.experiments.raw_support_closure' ;;
 *) exit 2 ;;
esac
