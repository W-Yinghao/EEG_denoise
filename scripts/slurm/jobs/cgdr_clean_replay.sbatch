#!/usr/bin/env bash
set -euo pipefail

readonly REPLAY_ROOT=/home/infres/yinwang/denoiseNet_replay_bc68f59
readonly MAIN_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly EXPECTED_HEAD=bc68f59663f6835aed12c4856d559da94eb39363
readonly OUTPUT_ROOT="$MAIN_ROOT/reports/cgdr_clean_replay_bc68f59"
readonly RUN_ROOT="$OUTPUT_ROOT/run"

[[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
    printf 'cgdr_clean_replay requires cpu\n' >&2
    exit 2
}
[[ -d "$REPLAY_ROOT/.git" || -f "$REPLAY_ROOT/.git" ]] || {
    printf 'missing clean replay worktree\n' >&2
    exit 2
}

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$OUTPUT_ROOT/pytest_tmp"
cd "$REPLAY_ROOT"

git rev-parse HEAD > "$OUTPUT_ROOT/git_head.txt"
git status --porcelain=v1 > "$OUTPUT_ROOT/git_status_before.txt"
git diff --no-ext-diff > "$OUTPUT_ROOT/git_diff_before.patch"
git diff --cached --no-ext-diff > "$OUTPUT_ROOT/git_diff_cached_before.patch"

[[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || {
    printf 'unexpected replay HEAD\n' >&2
    exit 2
}
[[ ! -s "$OUTPUT_ROOT/git_status_before.txt" ]] || {
    printf 'replay worktree is not clean\n' >&2
    exit 2
}

export PYTHONPATH="$REPLAY_ROOT/src:$REPLAY_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export HARNESS_LEVEL=1

"$EEG_ENV/bin/python" -m pytest -vv -p no:cacheprovider \
    --basetemp "$OUTPUT_ROOT/pytest_tmp" \
    tests/unit/test_cgdr_p0_precision.py \
    tests/unit/test_cgdr_pop_shrink.py \
    tests/unit/test_cgdr_sampler_semantics.py \
    tests/unit/test_cgdr_mechanism_data.py \
    tests/unit/test_cgdr_mechanism_aggregate.py \
    tests/unit/test_cgdr_mechanism_runner.py \
    2>&1 | tee "$OUTPUT_ROOT/pytest_full_output.txt"

"$EEG_ENV/bin/python" -m eeg_cgdr.cli.main mechanism-audit \
    --config configs/cgdr/mechanism_audit_klados.yaml \
    --run-dir "$RUN_ROOT" \
    --stage cpu-tests \
    2>&1 | tee "$OUTPUT_ROOT/validator_full_output.txt"

cp -- "$RUN_ROOT/mechanism_cpu_validation.json" \
    "$OUTPUT_ROOT/mechanism_cpu_validation.json"
git status --porcelain=v1 > "$OUTPUT_ROOT/git_status_after.txt"
git diff --no-ext-diff > "$OUTPUT_ROOT/git_diff_after.patch"
[[ ! -s "$OUTPUT_ROOT/git_status_after.txt" ]] || {
    printf 'replay worktree changed during validation\n' >&2
    exit 2
}
printf '%s\n' passed > "$OUTPUT_ROOT/status.txt"
