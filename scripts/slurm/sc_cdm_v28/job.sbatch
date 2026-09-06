#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_support_clean_cdm_v28
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
stage=${1:?stage required}
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_dir="$CODE_ROOT/results/sc_cdm_v28/runs/$stage/job_${SLURM_JOB_ID}"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_dir"
case "$stage" in r2-pareto|r4-sanity|r5-rounda|r6-rounda-eval|r9-*|r10-paired-infer|r11-natural-infer) env_root=$GPU_ENV;; *) env_root=$CPU_ENV;; esac
if [[ "$stage" == r16-tests ]]; then "$CPU_ENV/bin/python" -m pytest -q tests/unit/test_sc_cdm_v28.py | tee "$run_dir/pytest.txt"; exit 0; fi
if [[ "$stage" == r17-clean ]]; then
  clean_root=$(mktemp -d);trap 'rm -rf "$clean_root"' EXIT;git archive HEAD | tar -x -C "$clean_root";cd "$clean_root";export PYTHONPATH="$clean_root/src:$clean_root";"$CPU_ENV/bin/python" -c 'import eeg_scad.models.support_clean_cdm; import eeg_scad.cli.run_v28';"$CPU_ENV/bin/python" -m pytest -q tests/unit/test_sc_cdm_v28.py | tee "$run_dir/pytest.txt";exit 0
fi
exec "$env_root/bin/python" -m eeg_scad.cli.run_v28 --stage "$stage" --run-dir "$run_dir"
