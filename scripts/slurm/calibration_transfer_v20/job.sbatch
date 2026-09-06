#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_calibration_transfer_v20
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
stage=${1:?stage required}
cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_dir="$CODE_ROOT/results/cgdr/calibration_transfer_randomization_v20/runs/$stage/job_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
if [[ "$stage" == "p10-clean" ]]; then
  clean_root=$(mktemp -d)
  trap 'rm -rf "$clean_root"' EXIT
  git archive HEAD | tar -x -C "$clean_root"
  cd "$clean_root"
  export PYTHONPATH="$clean_root/src:$clean_root${PYTHONPATH:+:$PYTHONPATH}"
  "$CPU_ENV/bin/python" -c 'import eeg_cgdr.experiments.calibration_transfer_v20'
  "$CPU_ENV/bin/python" -m pytest -q tests/unit/test_calibration_transfer_v20.py | tee "$run_dir/pytest.txt"
  exit 0
fi
if [[ "$stage" == "p9-tests" ]]; then
  "$CPU_ENV/bin/python" -m pytest -q tests/unit/test_calibration_transfer_v20.py | tee "$run_dir/pytest.txt"
fi
exec "$CPU_ENV/bin/python" -m eeg_cgdr.cli.calibration_transfer_v20 \
  --config "$CODE_ROOT/configs/cgdr/calibration_transfer_randomization_v20.yaml" \
  --stage "$stage" --run-dir "$run_dir"

