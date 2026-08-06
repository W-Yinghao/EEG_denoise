#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_lit_explore_v3
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_dir="$CODE_ROOT/results/cgdr/literature_guided_exploration_v3/runs/j1-mobile-index/job_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
exec "$CPU_ENV/bin/python" -m eeg_cgdr.cli.mobile_bci_v3 --run-dir "$run_dir"
