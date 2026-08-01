#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly CONDA_BIN=/home/infres/yinwang/anaconda3/bin/conda
umask 077
cd "$CODE_ROOT"

payload_args_sha256=$(sha256sum /dev/null | awk '{print $1}')
[[ "$payload_args_sha256" == "${DENOISENET_PAYLOAD_ARGS_SHA256:?}" ]] || {
    printf 'validation payload arguments differ from the submission record\n' >&2
    exit 75
}

[[ $# -eq 0 ]] || { printf 'validate_control_plane accepts no arguments\n' >&2; exit 2; }
[[ "${DENOISENET_PROFILE:-}" == "cpu" ]] || {
    printf 'validate_control_plane requires the cpu profile\n' >&2
    exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == "CPU" ]] || {
    printf 'validate_control_plane requires the actual CPU partition\n' >&2
    exit 2
}
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'validate_control_plane does not support arrays\n' >&2
    exit 2
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
run_dir="$CODE_ROOT/reports/control_plane_validation/jobs/$job_id"
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
safe_ensure_code_directory "$CODE_ROOT/reports/control_plane_validation/jobs" || {
    printf 'unsafe validation output ancestry\n' >&2
    exit 1
}
[[ ! -e "$run_dir" && ! -L "$run_dir" ]] && mkdir -- "$run_dir" || {
    printf 'validation job directory already exists or cannot be created: %s\n' "$run_dir" >&2
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
    if [[ $rc -eq 0 ]]; then state=completed; elif [[ $rc -eq 75 ]]; then state=stale_input; else state=failed; fi
    cat > "$status_tmp" <<EOF
{
  "schema_version": 1,
  "job": "validate_control_plane",
  "job_id": "$job_id",
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
  "exit_code": $rc,
  "started_at_utc": "$started_at_utc",
  "finished_at_utc": "$finished_at_utc"
}
EOF
    publish_file_no_replace "$status_tmp" "$status_final" || rc=1
    exit "$rc"
}
trap finish EXIT

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
observed_job_script_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/jobs/validate_control_plane.sbatch" | awk '{print $1}')
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
[[ -f "$request_record" && ! -L "$request_record" ]] || exit 75
observed_request_hash=$(sha256sum "$request_record" | awk '{print $1}')
observed_git_head=$(git rev-parse --verify HEAD)
observed_git_branch=$(git rev-parse --abbrev-ref HEAD)
observed_git_origin=$(git remote get-url origin)
[[ "$observed_git_head" =~ ^[0-9a-f]{40,64}$ && \
   "$observed_git_branch" =~ ^[A-Za-z0-9._/-]+$ && \
   "$observed_git_origin" =~ ^https://github\.com/W-Yinghao/EEG_denoise(\.git)?$ ]] || exit 1
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

"$EEG_ENV/bin/python" "$CODE_ROOT/scripts/contract/validate_submission_request.py" \
    --output "$run_dir/submission-request-validation.json" \
    --expected-job validate_control_plane \
    --expected-profile cpu \
    --expected-partition CPU \
    --expected-dependency "" \
    --expected-payload-count 0 \
    --expected-payload-sha256 "$payload_args_sha256" \
    --expected-job-id "$job_id"

"$EEG_ENV/bin/python" "$CODE_ROOT/scripts/contract/capture_slurm_allocation.py" \
    --input "$run_dir/slurm_allocation.txt" \
    --output "$run_dir/slurm_allocation.json" \
    --expected-job-id "$job_id"

shell_files=("$CODE_ROOT/scripts/slurm/submit.sh")
while IFS= read -r -d '' shell_file; do
    shell_files+=("$shell_file")
done < <(find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0)
for shell_file in "${shell_files[@]}"; do
    /usr/bin/bash -n "$shell_file"
done

mapfile -d '' python_files < <(
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0
)
PYTHONPYCACHEPREFIX="$run_dir/pycache" \
    "$CONDA_BIN" run --no-capture-output -p "$EEG_ENV" \
    python -m py_compile "${python_files[@]}"
"$CONDA_BIN" run --no-capture-output -p "$EEG_ENV" \
    python "$CODE_ROOT/scripts/contract/validate_control_plane.py" \
    --code-root "$CODE_ROOT" \
    --output "$run_dir/validation.json"

sha256sum "$run_dir/validation.json" > "$run_dir/artifact-hashes.txt"

final_cluster_hash=$(sha256sum "$CODE_ROOT/configs/cluster/slurm.yaml" | awk '{print $1}')
final_environment_hash=$(sha256sum "$CODE_ROOT/configs/environments.yaml" | awk '{print $1}')
final_job_script_hash=$(sha256sum "$CODE_ROOT/scripts/slurm/jobs/validate_control_plane.sbatch" | awk '{print $1}')
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
    printf 'input hash changed during control-plane validation\n' >&2
    exit 75
fi
