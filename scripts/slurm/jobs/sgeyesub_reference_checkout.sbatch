#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EXTERNAL_ROOT="$CODE_ROOT/.external"
readonly TARGET="$EXTERNAL_ROOT/eyeartifactcorrection"
readonly SOURCE_URL=https://github.com/rkobler/eyeartifactcorrection.git
readonly EXPECTED_COMMIT=2c95b4f46f37670d25399ac0fdd705ae18248b25
readonly REPORT_ROOT="$CODE_ROOT/reports/sgeyesub_reference_checkout"
readonly job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}

[[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
    printf 'sgeyesub_reference_checkout requires the cpu profile\n' >&2
    exit 2
}
[[ $# -eq 0 ]] || {
    printf 'sgeyesub_reference_checkout accepts no arguments\n' >&2
    exit 2
}

cd "$CODE_ROOT"
umask 077
run_dir="$REPORT_ROOT/$job_id"
mkdir -p "$run_dir"
git rev-parse --verify HEAD > "$run_dir/git_head.txt"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"

actual_commit=unknown
origin_matches_expected=false
checkout_clean=false
key_files_present=false
expected_commit_reachable=false
published=false
checkout_mode=none

write_result() {
    local result_status=$1
    local checked_at result_tmp result_final clone_command
    checked_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    result_tmp="$run_dir/checkout.json.partial"
    result_final="$run_dir/checkout.json"
    clone_command="git clone --no-checkout $SOURCE_URL .external/.eyeartifactcorrection.partial-$job_id"
    cat > "$result_tmp" <<EOF
{
  "status": "$result_status",
  "slurm_job_id": "$job_id",
  "profile": "cpu",
  "checked_at_utc": "$checked_at",
  "source_repository": "$SOURCE_URL",
  "expected_commit": "$EXPECTED_COMMIT",
  "actual_commit": "$actual_commit",
  "official_code_root": "$TARGET",
  "checkout_mode": "$checkout_mode",
  "origin_matches_expected": $origin_matches_expected,
  "expected_commit_reachable": $expected_commit_reachable,
  "checkout_clean": $checkout_clean,
  "key_files_present": $key_files_present,
  "published": $published,
  "clone_command": "$clone_command",
  "overwrite_policy": "existing target is inspected read-only and never fetched, checked out, or overwritten"
}
EOF
    mv -- "$result_tmp" "$result_final"
    printf 'status=%s\n' "$result_status"
    printf 'actual_commit=%s\n' "$actual_commit"
    printf 'result=%s\n' "$result_final"
}

inspect_checkout() {
    local checkout_path=$1
    local origin_url candidate_commit status_output required_file
    actual_commit=unknown
    origin_matches_expected=false
    checkout_clean=false
    key_files_present=false
    expected_commit_reachable=false

    [[ -d "$checkout_path" && ! -L "$checkout_path" ]] || return 1
    git -C "$checkout_path" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1

    origin_url=$(git -C "$checkout_path" remote get-url origin 2>/dev/null || true)
    case "$origin_url" in
        https://github.com/rkobler/eyeartifactcorrection|https://github.com/rkobler/eyeartifactcorrection.git)
            origin_matches_expected=true
            ;;
    esac

    candidate_commit=$(git -C "$checkout_path" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ "$candidate_commit" =~ ^[0-9a-f]{40}$ ]]; then
        actual_commit=$candidate_commit
    fi
    if git -C "$checkout_path" cat-file -e "$EXPECTED_COMMIT^{commit}" 2>/dev/null; then
        expected_commit_reachable=true
    fi

    status_output=$(git -C "$checkout_path" status --porcelain --untracked-files=all 2>/dev/null || true)
    if [[ -z "$status_output" ]]; then
        checkout_clean=true
    fi
    unset status_output

    key_files_present=true
    for required_file in \
        algorithms/sgeyesub.m \
        demo_main.m \
        utility/equalizeLabels.m \
        external/cov_shrink.m
    do
        if [[ ! -f "$checkout_path/$required_file" || -L "$checkout_path/$required_file" ]]; then
            key_files_present=false
        fi
    done
    return 0
}

if [[ -L "$EXTERNAL_ROOT" ]]; then
    write_result blocked_unsafe_external_root
    exit 0
fi
mkdir -p "$EXTERNAL_ROOT"
if [[ ! -d "$EXTERNAL_ROOT" || -L "$EXTERNAL_ROOT" \
    || "$(realpath -e -- "$EXTERNAL_ROOT")" != "$EXTERNAL_ROOT" ]]; then
    write_result blocked_unsafe_external_root
    exit 0
fi

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
    checkout_mode=existing_read_only
    if ! inspect_checkout "$TARGET"; then
        write_result blocked_existing_target_not_checkout
        exit 0
    fi
    if [[ "$origin_matches_expected" == true \
        && "$actual_commit" == "$EXPECTED_COMMIT" \
        && "$checkout_clean" == true \
        && "$key_files_present" == true ]]; then
        write_result verified_existing_frozen_checkout
    else
        write_result blocked_existing_checkout_mismatch
    fi
    exit 0
fi

partial="$EXTERNAL_ROOT/.eyeartifactcorrection.partial-$job_id"
if [[ -e "$partial" || -L "$partial" ]]; then
    write_result blocked_partial_path_already_exists
    exit 0
fi

checkout_mode=cloned_by_job
set +e
git clone --no-checkout "$SOURCE_URL" "$partial"
clone_rc=$?
set -e
if [[ $clone_rc -ne 0 ]]; then
    write_result blocked_clone_failed
    exit 0
fi

if ! inspect_checkout "$partial"; then
    write_result blocked_cloned_directory_not_checkout
    exit 0
fi
if [[ "$origin_matches_expected" != true || "$expected_commit_reachable" != true ]]; then
    write_result blocked_expected_commit_unavailable
    exit 0
fi

set +e
git -C "$partial" checkout --detach "$EXPECTED_COMMIT"
checkout_rc=$?
set -e
if [[ $checkout_rc -ne 0 ]]; then
    inspect_checkout "$partial" || true
    write_result blocked_expected_commit_checkout_failed
    exit 0
fi
inspect_checkout "$partial" || {
    write_result blocked_checked_out_directory_invalid
    exit 0
}
if [[ "$actual_commit" != "$EXPECTED_COMMIT" \
    || "$checkout_clean" != true \
    || "$key_files_present" != true ]]; then
    write_result blocked_checked_out_content_mismatch
    exit 0
fi

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
    checkout_mode=concurrent_target_read_only
    inspect_checkout "$TARGET" || true
    write_result blocked_concurrent_target_appeared
    exit 0
fi

# GNU mv -T -n keeps publication on this filesystem atomic and refuses to
# replace a target that appears during the final race window.
mv -T -n -- "$partial" "$TARGET"
if [[ -e "$partial" || -L "$partial" ]]; then
    checkout_mode=concurrent_target_read_only
    inspect_checkout "$TARGET" || true
    write_result blocked_concurrent_target_appeared
    exit 0
fi
published=true
checkout_mode=published_by_job
inspect_checkout "$TARGET" || {
    write_result blocked_published_checkout_invalid
    exit 0
}
if [[ "$origin_matches_expected" == true \
    && "$actual_commit" == "$EXPECTED_COMMIT" \
    && "$checkout_clean" == true \
    && "$key_files_present" == true ]]; then
    write_result verified_frozen_checkout
else
    write_result blocked_published_checkout_mismatch
fi
