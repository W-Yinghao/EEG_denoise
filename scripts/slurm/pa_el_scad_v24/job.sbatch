#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_pa_el_scad_v24
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
stage=${1:?stage required}
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_dir="$CODE_ROOT/results/pa_el_scad_v24/runs/$stage/job_${SLURM_JOB_ID}"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_dir"
case "$stage" in
  r4-sanity|r5-anchor|r6-det|r7-diff|r8-eval|r10-anchor|r10-det|r10-diff|r11-paired|r12-natural-infer|r15-optional) env_root="$GPU_ENV" ;;
  *) env_root="$CPU_ENV" ;;
esac
if [[ "$stage" == r17-tests ]]; then
  "$CPU_ENV/bin/python" -m pytest -q tests/unit/test_pa_el_scad_v24.py | tee "$run_dir/pytest.txt"
  exit 0
fi
if [[ "$stage" == r18-clean ]]; then
  clean_root=$(mktemp -d)
  trap 'rm -rf "$clean_root"' EXIT
  git archive HEAD | tar -x -C "$clean_root"
  cd "$clean_root"
  export PYTHONPATH="$clean_root/src:$clean_root${PYTHONPATH:+:$PYTHONPATH}"
  "$CPU_ENV/bin/python" -c 'import eeg_scad; import eeg_scad.cli.v24'
  "$CPU_ENV/bin/python" -m pytest -q tests/unit/test_pa_el_scad_v24.py | tee "$run_dir/pytest.txt"
  exit 0
fi
exec "$env_root/bin/python" -m eeg_scad.cli.v24 --stage "$stage" --run-dir "$run_dir"
