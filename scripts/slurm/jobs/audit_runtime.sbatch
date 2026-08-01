#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly CONDA_BIN=/home/infres/yinwang/anaconda3/bin/conda
readonly RENDERER_PROBE="$CODE_ROOT/scripts/contract/probe_renderer_import.py"
umask 077
ulimit -c 0
cd "$CODE_ROOT"
original_args=("$@")
payload_args_sha256=$(
    {
        if [[ ${#original_args[@]} -gt 0 ]]; then
            printf '%s\0' "${original_args[@]}"
        fi
    } | sha256sum | awk '{print $1}'
)
[[ "$payload_args_sha256" == "${DENOISENET_PAYLOAD_ARGS_SHA256:?}" ]] || {
    printf 'runtime-audit payload arguments differ from the submission record\n' >&2
    exit 75
}

env_name=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            [[ $# -ge 2 ]] || { printf 'missing value for --env\n' >&2; exit 2; }
            env_name=$2
            shift 2
            ;;
        *)
            printf 'unexpected argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

case "$env_name" in
    eeg2025) env_path=/home/infres/yinwang/anaconda3/envs/eeg2025 ;;
    icml) env_path=/home/infres/yinwang/anaconda3/envs/icml ;;
    *) printf 'unregistered environment: %s\n' "$env_name" >&2; exit 2 ;;
esac

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
output_root="$CODE_ROOT/reports/environments/$env_name"
run_dir="$output_root/jobs/$job_id"
safe_ensure_code_directory() {
    local target=$1
    local relative current component
    [[ "$target" == "$CODE_ROOT"/* ]] || return 1
    [[ "$(realpath -e -- "$CODE_ROOT")" == "$CODE_ROOT" ]] || return 1
    relative=${target#"$CODE_ROOT"/}
    current=$CODE_ROOT
    IFS='/' read -r -a components <<< "$relative"
    for component in "${components[@]}"; do
        [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
        current="$current/$component"
        if [[ -e "$current" || -L "$current" ]]; then
            [[ -d "$current" && ! -L "$current" ]] || return 1
        else
            mkdir -- "$current" || return 1
        fi
        [[ "$(realpath -e -- "$current")" == "$current" ]] || return 1
    done
}
publish_file_no_replace() {
    local source=$1
    local target=$2
    [[ -f "$source" && ! -L "$source" && ! -e "$target" && ! -L "$target" ]] || return 1
    ln -T -- "$source" "$target" || return 1
    rm -- "$source" || return 1
    [[ ! -e "$source" && -f "$target" && ! -L "$target" ]]
}
safe_ensure_code_directory "$output_root/jobs" || {
    printf 'unsafe runtime-audit output ancestry\n' >&2
    exit 1
}
[[ ! -e "$run_dir" && ! -L "$run_dir" ]] && mkdir -- "$run_dir" || {
    printf 'runtime-audit job directory already exists or cannot be created: %s\n' "$run_dir" >&2
    exit 1
}
[[ "$(realpath -e -- "$run_dir")" == "$run_dir" ]] || exit 1
set -o noclobber

started_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
status_tmp="$run_dir/status.json.partial"
status_final="$run_dir/status.json"
finish() {
    rc=$?
    trap - EXIT
    finished_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then
        state=completed
        provenance_complete=true
    elif [[ $rc -eq 75 ]]; then
        state=stale_input
        provenance_complete=false
    else
        state=failed
        provenance_complete=false
    fi
    cat > "$status_tmp" <<EOF
{
  "schema_version": 1,
  "job": "audit_runtime",
  "job_id": "$job_id",
  "environment_name": "$env_name",
  "environment_path": "$env_path",
  "profile": "${DENOISENET_PROFILE:-unknown}",
  "request_id": "${DENOISENET_REQUEST_ID:-unknown}",
  "request_sha256": "${DENOISENET_REQUEST_SHA256:-unknown}",
  "cluster_config_sha256": "${DENOISENET_SUBMIT_CONFIG_SHA256:-unknown}",
  "environment_config_sha256": "${DENOISENET_ENV_CONFIG_SHA256:-unknown}",
  "job_script_sha256": "${DENOISENET_JOB_SCRIPT_SHA256:-unknown}",
  "submitter_sha256": "${DENOISENET_SUBMITTER_SHA256:-unknown}",
  "contract_bundle_sha256": "${DENOISENET_CONTRACT_BUNDLE_SHA256:-unknown}",
  "slurm_jobs_bundle_sha256": "${DENOISENET_SLURM_JOBS_BUNDLE_SHA256:-unknown}",
  "payload_arguments_sha256": "${DENOISENET_PAYLOAD_ARGS_SHA256:-unknown}",
  "git_commit": "${observed_git_head:-unknown}",
  "git_branch_or_detached": "${observed_git_branch:-unknown}",
  "git_remotes_sha256": "${observed_git_remotes_hash:-unknown}",
  "git_origin": "${observed_git_origin:-unknown}",
  "tracked_worktree_diff_sha256": "${observed_git_diff_hash:-unknown}",
  "untracked_path_list_sha256": "${observed_untracked_paths_hash:-unknown}",
  "state": "$state",
  "provenance_complete": $provenance_complete,
  "exit_code": $rc,
  "started_at_utc": "$started_at_utc",
  "finished_at_utc": "$finished_at_utc"
}
EOF
    publish_file_no_replace "$status_tmp" "$status_final" || rc=1
    exit "$rc"
}
trap finish EXIT

[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'audit_runtime does not support job arrays\n' >&2
    exit 2
}
case "$env_name:${DENOISENET_PROFILE:-}" in
    eeg2025:cpu)
        expected_partition=CPU
        dependency_policy=(--expected-dependency "")
        ;;
    icml:L40S)
        expected_partition=L40S
        dependency_policy=(--dependency-pattern 'afterok:[0-9]+')
        ;;
    *) printf 'environment/profile pairing is not registered for runtime audit\n' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == "$expected_partition" ]] || {
    printf 'actual Slurm partition differs from registered runtime-audit partition\n' >&2
    exit 2
}

{
    printf '%q ' "$0" "${original_args[@]}"
    printf '\n'
} > "$run_dir/invocation.txt"
safe_slurm_variables=(
    SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST
    SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_NUM_NODES SLURM_NTASKS
    SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_RESTART_COUNT SLURMD_NODENAME
)
exec 8> "$run_dir/slurm_environment.txt"
for variable_name in "${safe_slurm_variables[@]}"; do
    if [[ -v "$variable_name" ]]; then
        printf '%s=%q\n' "$variable_name" "${!variable_name}" \
            >&8
    fi
done
exec 8>&-
scontrol show job -o "$job_id" > "$run_dir/slurm_allocation.txt" 2> "$run_dir/slurm_allocation.stderr"

observed_cluster_hash=$(sha256sum "$CODE_ROOT/configs/cluster/slurm.yaml" | awk '{print $1}')
observed_environment_hash=$(sha256sum "$CODE_ROOT/configs/environments.yaml" | awk '{print $1}')
observed_job_script_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/jobs/audit_runtime.sbatch" | awk '{print $1}')
observed_submitter_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/submit.sh" | awk '{print $1}')
observed_contract_bundle_hash=$(
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
observed_slurm_jobs_bundle_hash=$(
    find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
request_record="$CODE_ROOT/reports/slurm/submissions/requests/${DENOISENET_REQUEST_ID:?}.json"
[[ -f "$request_record" && ! -L "$request_record" ]] || {
    printf 'submission request record is absent or unsafe\n' >&2
    exit 75
}
observed_request_hash=$(sha256sum "$request_record" | awk '{print $1}')
observed_git_head=$(git rev-parse --verify HEAD)
observed_git_branch=$(git rev-parse --abbrev-ref HEAD)
observed_git_origin=$(git remote get-url origin)
[[ "$observed_git_head" =~ ^[0-9a-f]{40,64}$ && \
   "$observed_git_branch" =~ ^[A-Za-z0-9._/-]+$ && \
   "$observed_git_origin" =~ ^https://github\.com/W-Yinghao/EEG_denoise(\.git)?$ ]] || {
    printf 'unsafe Git commit or branch provenance\n' >&2
    exit 1
}
observed_git_remotes_hash=$(git remote -v | sha256sum | awk '{print $1}')
observed_git_diff_hash=$(git diff --binary HEAD | sha256sum | awk '{print $1}')
observed_untracked_paths_hash=$(git ls-files --others --exclude-standard -z | sha256sum | awk '{print $1}')
printf '%s\n' "$observed_git_head" > "$run_dir/git-commit.txt"
printf '%s\n' "$observed_git_branch" > "$run_dir/git-branch.txt"
printf '%s\n' "$observed_git_origin" > "$run_dir/git-origin.txt"
printf '%s\n' "$observed_git_remotes_hash" > "$run_dir/git-remotes.sha256"
printf '%s\n' "$observed_git_diff_hash" > "$run_dir/tracked-worktree-diff.sha256"
printf '%s\n' "$observed_untracked_paths_hash" > "$run_dir/untracked-path-list.sha256"
if [[ "$observed_cluster_hash" != "${DENOISENET_SUBMIT_CONFIG_SHA256:?}" || \
      "$observed_environment_hash" != "${DENOISENET_ENV_CONFIG_SHA256:?}" || \
      "$observed_job_script_hash" != "${DENOISENET_JOB_SCRIPT_SHA256:?}" || \
      "$observed_submitter_hash" != "${DENOISENET_SUBMITTER_SHA256:?}" || \
      "$observed_contract_bundle_hash" != "${DENOISENET_CONTRACT_BUNDLE_SHA256:?}" || \
      "$observed_slurm_jobs_bundle_hash" != "${DENOISENET_SLURM_JOBS_BUNDLE_SHA256:?}" || \
      "$observed_request_hash" != "${DENOISENET_REQUEST_SHA256:?}" ]]; then
    printf 'input hash changed between submission and execution\n' >&2
    exit 75
fi

[[ -x "$CONDA_BIN" ]] || { printf 'missing conda executable: %s\n' "$CONDA_BIN" >&2; exit 1; }
[[ -x "$env_path/bin/python" && -d "$env_path/conda-meta" ]] || {
    printf 'registered environment is incomplete: %s\n' "$env_path" >&2
    exit 1
}
"$env_path/bin/python" "$CODE_ROOT/scripts/contract/validate_submission_request.py" \
    --output "$run_dir/submission-request-validation.json" \
    --expected-job audit_runtime \
    --expected-profile "${DENOISENET_PROFILE}" \
    --expected-partition "$expected_partition" \
    "${dependency_policy[@]}" \
    --expected-payload-count 2 \
    --expected-payload-sha256 "$payload_args_sha256" \
    --expected-job-id "$job_id"
"$env_path/bin/python" "$CODE_ROOT/scripts/contract/capture_slurm_allocation.py" \
    --input "$run_dir/slurm_allocation.txt" \
    --output "$run_dir/slurm_allocation.json" \
    --expected-job-id "$job_id"

set +e
"$env_path/bin/python" "$CODE_ROOT/scripts/contract/capture_sanitized_command.py" \
    --stdout-out "$run_dir/conda-explicit.txt" \
    --stderr-out "$run_dir/conda-explicit.stderr" \
    --audit-out "$run_dir/conda-explicit-sanitization.json" \
    --stdout-format conda-explicit \
    -- "$CONDA_BIN" list --explicit -p "$env_path"
conda_rc=$?
"$env_path/bin/python" "$CODE_ROOT/scripts/contract/capture_sanitized_command.py" \
    --stdout-out "$run_dir/pip-freeze.txt" \
    --stderr-out "$run_dir/pip-freeze.stderr" \
    --audit-out "$run_dir/pip-freeze-sanitization.json" \
    --stdout-format pip-freeze \
    -- "$CONDA_BIN" run --no-capture-output -p "$env_path" python -m pip freeze
pip_rc=$?
"$CONDA_BIN" run --no-capture-output -p "$env_path" python --version > "$run_dir/python-version.txt" 2>&1
python_version_rc=$?
set -e

printf '%s\n' "$conda_rc" > "$run_dir/conda-explicit.exit-code"
printf '%s\n' "$pip_rc" > "$run_dir/pip-freeze.exit-code"
printf '%s\n' "$python_version_rc" > "$run_dir/python-version.exit-code"
[[ $conda_rc -eq 0 && $pip_rc -eq 0 && $python_version_rc -eq 0 ]] || exit 1

sha256sum "$run_dir/conda-explicit.txt" > "$run_dir/conda-explicit.sha256"
sha256sum "$run_dir/pip-freeze.txt" > "$run_dir/pip-freeze.sha256"

"$CONDA_BIN" run --no-capture-output -p "$env_path" \
    python "$CODE_ROOT/scripts/contract/runtime_probe.py" \
    --env-name "$env_name" \
    --output "$run_dir/runtime_probe.json"

if [[ "$env_name" == icml ]]; then
    set +e
    PYTHONWARNINGS=error "$env_path/bin/python" "$RENDERER_PROBE" \
        --mode direct \
        --output "$run_dir/renderer-import-direct.json" \
        > /dev/null 2> /dev/null
    renderer_direct_rc=$?
    PYTHONWARNINGS=error "$CONDA_BIN" run --no-capture-output -p "$env_path" \
        python "$RENDERER_PROBE" \
        --mode conda_run \
        --output "$run_dir/renderer-import-conda_run.json" \
        > /dev/null 2> /dev/null
    renderer_conda_rc=$?
    set -e
    {
        printf '{\n'
        printf '  "schema_version": 1,\n'
        printf '  "job_id": "%s",\n' "$job_id"
        printf '  "direct_exit_code": %s,\n' "$renderer_direct_rc"
        printf '  "conda_run_exit_code": %s,\n' "$renderer_conda_rc"
        printf '  "core_dump_limit": 0,\n'
        printf '  "registered_renderer_mode": "conda_run"\n'
        printf '}\n'
    } > "$run_dir/renderer-import-comparison.json"
    [[ $renderer_conda_rc -eq 0 && -f "$run_dir/renderer-import-conda_run.json" ]] || {
        printf 'registered conda-run renderer cold-start probe failed\n' >&2
        exit 1
    }
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    set +e
    nvidia-smi -L > "$run_dir/nvidia-smi-L.txt" 2> "$run_dir/nvidia-smi-L.stderr"
    smi_list_rc=$?
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
        > "$run_dir/nvidia-smi-query.csv" 2> "$run_dir/nvidia-smi-query.stderr"
    smi_query_rc=$?
    set -e
    printf '%s\n' "$smi_list_rc" > "$run_dir/nvidia-smi-L.exit-code"
    printf '%s\n' "$smi_query_rc" > "$run_dir/nvidia-smi-query.exit-code"
    if [[ "$env_name" == "icml" && ( $smi_list_rc -ne 0 || $smi_query_rc -ne 0 ) ]]; then
        exit 1
    fi
else
    printf 'nvidia-smi unavailable\n' > "$run_dir/nvidia-smi-L.stderr"
    [[ "$env_name" != "icml" ]] || exit 1
fi

sha256sum "$CODE_ROOT/configs/cluster/slurm.yaml" "$CODE_ROOT/configs/environments.yaml" \
    > "$run_dir/config-hashes.txt"

"$env_path/bin/python" "$CODE_ROOT/scripts/contract/capture_sanitized_command.py" \
    --stdout-out "$run_dir/conda-explicit-end.txt" \
    --stderr-out "$run_dir/conda-explicit-end.stderr" \
    --audit-out "$run_dir/conda-explicit-end-sanitization.json" \
    --stdout-format conda-explicit \
    -- "$CONDA_BIN" list --explicit -p "$env_path"
"$env_path/bin/python" "$CODE_ROOT/scripts/contract/capture_sanitized_command.py" \
    --stdout-out "$run_dir/pip-freeze-end.txt" \
    --stderr-out "$run_dir/pip-freeze-end.stderr" \
    --audit-out "$run_dir/pip-freeze-end-sanitization.json" \
    --stdout-format pip-freeze \
    -- "$CONDA_BIN" run --no-capture-output -p "$env_path" python -m pip freeze
if ! cmp -- "$run_dir/conda-explicit.txt" "$run_dir/conda-explicit-end.txt" || \
   ! cmp -- "$run_dir/conda-explicit-sanitization.json" "$run_dir/conda-explicit-end-sanitization.json" || \
   ! cmp -- "$run_dir/pip-freeze.txt" "$run_dir/pip-freeze-end.txt" || \
   ! cmp -- "$run_dir/pip-freeze-sanitization.json" "$run_dir/pip-freeze-end-sanitization.json"; then
    printf 'registered environment changed during runtime audit\n' >&2
    exit 75
fi

final_cluster_hash=$(sha256sum "$CODE_ROOT/configs/cluster/slurm.yaml" | awk '{print $1}')
final_environment_hash=$(sha256sum "$CODE_ROOT/configs/environments.yaml" | awk '{print $1}')
final_job_script_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/jobs/audit_runtime.sbatch" | awk '{print $1}')
final_submitter_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/submit.sh" | awk '{print $1}')
final_contract_bundle_hash=$(
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
final_slurm_jobs_bundle_hash=$(
    find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
final_request_hash=$(sha256sum "$request_record" | awk '{print $1}')
final_git_head=$(git rev-parse --verify HEAD)
final_git_branch=$(git rev-parse --abbrev-ref HEAD)
final_git_origin=$(git remote get-url origin)
final_git_remotes_hash=$(git remote -v | sha256sum | awk '{print $1}')
final_git_diff_hash=$(git diff --binary HEAD | sha256sum | awk '{print $1}')
final_untracked_paths_hash=$(git ls-files --others --exclude-standard -z | sha256sum | awk '{print $1}')
if [[ "$final_cluster_hash" != "$observed_cluster_hash" || \
      "$final_environment_hash" != "$observed_environment_hash" || \
      "$final_job_script_hash" != "$observed_job_script_hash" || \
      "$final_submitter_hash" != "$observed_submitter_hash" || \
      "$final_contract_bundle_hash" != "$observed_contract_bundle_hash" || \
      "$final_slurm_jobs_bundle_hash" != "$observed_slurm_jobs_bundle_hash" || \
      "$final_request_hash" != "$observed_request_hash" || \
      "$final_git_head" != "$observed_git_head" || \
      "$final_git_branch" != "$observed_git_branch" || \
      "$final_git_origin" != "$observed_git_origin" || \
      "$final_git_remotes_hash" != "$observed_git_remotes_hash" || \
      "$final_git_diff_hash" != "$observed_git_diff_hash" || \
      "$final_untracked_paths_hash" != "$observed_untracked_paths_hash" ]]; then
    printf 'input hash changed during runtime audit\n' >&2
    exit 75
fi
