#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_lit_explore_v3
readonly CPU_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml

stage=${1:?stage is required}
config=${2:-configs/cgdr/literature_guided_exploration_v3/exploration.yaml}
profile=${DENOISENET_PROFILE:?DENOISENET_PROFILE is required}
case "$stage" in
    j0-audit|j0-mobile-download|j1-tasks|j1-mobile-index|j5-selector|j7-aggregate|j8-final)
        python_bin="$CPU_ENV/bin/python"
        ;;
    j2-tech|j2-baselines|j3-raw-support|j4-lora|j6-control)
        python_bin="$GPU_ENV/bin/python"
        ;;
    *)
        printf 'unsupported literature-v3 stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

cd "$CODE_ROOT"
export DENOISENET_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DENOISENET_SGE_STRUCTURE_AUDIT=/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919218/attempt-0/result.json
export DENOISENET_GIT_HEAD
DENOISENET_GIT_HEAD=$(git rev-parse HEAD)
run_dir="$CODE_ROOT/results/cgdr/literature_guided_exploration_v3/runs/${stage}/job_${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_dir="$run_dir/task_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$run_dir"

if [[ "$stage" == j0-audit || "$stage" == j1-tasks || "$stage" == j8-final ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_literature_guided_v3.py \
        | tee "$run_dir/pytest.txt"
fi

if [[ "$stage" == j8-final ]]; then
    clean_checkout=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/litv3-clean-checkout.XXXXXX")
    git archive HEAD | tar -x -C "$clean_checkout"
    env PYTHONPATH="$clean_checkout/src:$clean_checkout" "$python_bin" - <<'PY' | tee "$run_dir/clean_checkout_import.txt"
import eeg_cgdr
from eeg_cgdr.experiments import literature_guided_v3
from eeg_cgdr.experiments import official_baselines_v3, selective_policy_v3
print("clean_checkout_import=passed")
print(f"protocol={literature_guided_v3.PROTOCOL}")
PY
fi

if [[ "$stage" == j2-baselines ]]; then
    exec "$python_bin" -m eeg_cgdr.cli.official_baselines_v3 --run-dir "$run_dir"
fi

if [[ "$stage" == j5-selector ]]; then
    exec "$python_bin" -m eeg_cgdr.cli.selective_policy_v3 \
        --config "$CODE_ROOT/$config" --run-dir "$run_dir"
fi

if [[ "$stage" == j1-mobile-index ]]; then
    exec "$python_bin" -m eeg_cgdr.cli.mobile_bci_v3 --run-dir "$run_dir"
fi

exec "$python_bin" -m eeg_cgdr.cli.literature_guided_v3 \
    --config "$CODE_ROOT/$config" \
    --stage "$stage" \
    --run-dir "$run_dir"
