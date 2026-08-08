#!/usr/bin/env bash
#SBATCH --account=c2s
#SBATCH --qos=normal
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_basis_score_v9r
readonly CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
readonly GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
stage=${1:?stage};cd "$ROOT";export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
array_job_id=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID};run="$ROOT/results/cgdr/sge_basis_score_factorial_v9r/runs/${stage}/job_${array_job_id}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}";mkdir -p "$run"
case "$stage" in
  tests) "$CPU" -m pytest -q tests/unit/test_sge_basis_score_factorial_v9r.py tests/unit/test_sge_basis_score_factorial_v9.py tests/unit/test_sge_eb_bridge_v8_1.py | tee "$run/pytest.txt" ;;
  clean-import) clean_root="${SLURM_TMPDIR:-$run/clean-checkout}";mkdir -p "$clean_root";git archive HEAD | tar -x -C "$clean_root";PYTHONPATH="$clean_root/src:$clean_root" "$CPU" -c 'import eeg_cgdr.experiments.sge_basis_score_factorial_v9r; import eeg_cgdr.models.adaptation_replay; print("V9R clean-checkout import passed")' | tee "$run/clean_import.txt" ;;
  technical|exact-infer) exec "$GPU" -m eeg_cgdr.cli.sge_basis_score_factorial_v9r --config configs/cgdr/sge_basis_score_factorial_v9r.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  v9-audit|coverage|coverage-aggregate|exact-build|exact-eval|exact-aggregate|finalize) exec "$CPU" -m eeg_cgdr.cli.sge_basis_score_factorial_v9r --config configs/cgdr/sge_basis_score_factorial_v9r.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
  *) exit 2 ;;
esac
