#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly DATA_ROOT=/projects/EEG-foundation-model
readonly OUTPUT_PARENT="$CODE_ROOT/reports/data_inventory"
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly CONDA_BIN=/home/infres/yinwang/anaconda3/bin/conda
readonly CLUSTER_CONFIG="$CODE_ROOT/configs/cluster/slurm.yaml"
readonly ENVIRONMENT_CONFIG="$CODE_ROOT/configs/environments.yaml"
readonly JOB_SCRIPT="$CODE_ROOT/scripts/slurm/jobs/inventory_data.sbatch"
readonly SCANNER="$CODE_ROOT/scripts/contract/inventory_data.py"
readonly SUBMITTER="$CODE_ROOT/scripts/slurm/submit.sh"
umask 077
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
    printf 'inventory payload arguments differ from the submission record\n' >&2
    exit 75
}
data_root=""
output_root=""
runtime_audit_job=""
read_only=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            [[ $# -ge 2 ]] || { printf 'missing value for --data-root\n' >&2; exit 2; }
            data_root=$2
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { printf 'missing value for --output-root\n' >&2; exit 2; }
            output_root=$2
            shift 2
            ;;
        --runtime-audit-job)
            [[ $# -ge 2 ]] || { printf 'missing value for --runtime-audit-job\n' >&2; exit 2; }
            runtime_audit_job=$2
            shift 2
            ;;
        --read-only)
            read_only=true
            shift
            ;;
        *)
            printf 'unexpected argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

[[ "$data_root" == "$DATA_ROOT" ]] || {
    printf 'data root must be exactly %s\n' "$DATA_ROOT" >&2
    exit 2
}
case "$output_root" in
    reports/data_inventory|"$OUTPUT_PARENT") ;;
    *) printf 'output root must be reports/data_inventory\n' >&2; exit 2 ;;
esac
[[ "$runtime_audit_job" =~ ^[0-9]+$ ]] || {
    printf 'a numeric --runtime-audit-job is required\n' >&2
    exit 2
}
[[ "$read_only" == true ]] || {
    printf -- '--read-only is mandatory\n' >&2
    exit 2
}
[[ "${DENOISENET_PROFILE:?}" == "cpu-high" && "${DENOISENET_JOB:?}" == "inventory_data" ]] || {
    printf 'inventory_data must use the registered cpu-high submitter profile\n' >&2
    exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == "cpu-high" ]] || {
    printf 'actual Slurm partition is not cpu-high\n' >&2
    exit 2
}
[[ -x "$CONDA_BIN" && -x "$EEG_ENV/bin/python" ]] || {
    printf 'registered eeg2025 environment is unavailable\n' >&2
    exit 1
}
for required_file in "$CLUSTER_CONFIG" "$ENVIRONMENT_CONFIG" "$JOB_SCRIPT" "$SCANNER"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] || {
        printf 'required input is missing or unsafe: %s\n' "$required_file" >&2
        exit 1
    }
done
[[ -f "$SUBMITTER" && ! -L "$SUBMITTER" ]] || {
    printf 'submitter is missing or unsafe\n' >&2
    exit 1
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
restart_count=${SLURM_RESTART_COUNT:-0}
[[ "$job_id" =~ ^[0-9]+$ && "$restart_count" =~ ^[0-9]+$ ]] || {
    printf 'invalid Slurm job or restart identifier\n' >&2
    exit 1
}
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'inventory_data does not support job arrays\n' >&2
    exit 2
}

attempt_dir="$OUTPUT_PARENT/jobs/$job_id/attempt-$restart_count"
scan_dir="$attempt_dir/scan"

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

safe_ensure_code_directory "$OUTPUT_PARENT/jobs/$job_id" || {
    printf 'unsafe inventory output ancestry; refusing to create job output\n' >&2
    exit 1
}
[[ ! -e "$attempt_dir" && ! -L "$attempt_dir" ]] && mkdir -- "$attempt_dir" || {
    printf 'inventory attempt directory already exists or cannot be created: %s\n' "$attempt_dir" >&2
    exit 1
}
[[ "$(realpath -e -- "$attempt_dir")" == "$attempt_dir" ]] || exit 1
set -o noclobber
[[ ! -e "$scan_dir" && ! -L "$scan_dir" ]] || {
    printf 'scan output already exists; refusing to overwrite: %s\n' "$scan_dir" >&2
    exit 1
}

started_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
job_state=failed
scan_exit_code=1
status_tmp="$attempt_dir/job_status.json.partial"
status_final="$attempt_dir/job_status.json"
finish() {
    rc=$?
    trap - EXIT
    finished_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    case "$rc" in
        0) job_state=completed ;;
        4) job_state=partial ;;
        75) job_state=stale_input ;;
        *) job_state=failed ;;
    esac
    cat > "$status_tmp" <<EOF
{
  "schema_version": 1,
  "job": "inventory_data",
  "job_id": "$job_id",
  "attempt": $restart_count,
  "profile": "${DENOISENET_PROFILE:-unknown}",
  "runtime_audit_job_id": "$runtime_audit_job",
  "request_id": "${DENOISENET_REQUEST_ID:-unknown}",
  "request_sha256": "${DENOISENET_REQUEST_SHA256:-unknown}",
  "cluster_config_sha256": "${DENOISENET_SUBMIT_CONFIG_SHA256:-unknown}",
  "environment_config_sha256": "${DENOISENET_ENV_CONFIG_SHA256:-unknown}",
  "job_script_sha256": "${DENOISENET_JOB_SCRIPT_SHA256:-unknown}",
  "contract_bundle_sha256": "${DENOISENET_CONTRACT_BUNDLE_SHA256:-unknown}",
  "slurm_jobs_bundle_sha256": "${DENOISENET_SLURM_JOBS_BUNDLE_SHA256:-unknown}",
  "payload_arguments_sha256": "${DENOISENET_PAYLOAD_ARGS_SHA256:-unknown}",
  "state": "$job_state",
  "exit_code": $rc,
  "scanner_exit_code": $scan_exit_code,
  "started_at_utc": "$started_at_utc",
  "finished_at_utc": "$finished_at_utc",
  "scan_status_path": "$scan_dir/status.json"
}
EOF
    publish_file_no_replace "$status_tmp" "$status_final" || rc=1
    if [[ $rc -ne 0 && ! -e "$scan_dir/PARTIAL" && ! -L "$scan_dir/PARTIAL" ]]; then
        fallback_marker_tmp="$attempt_dir/PARTIAL.partial"
        fallback_marker="$attempt_dir/PARTIAL"
        cat > "$fallback_marker_tmp" <<EOF
{
  "schema_version": 1,
  "state": "PARTIAL",
  "job_id": "$job_id",
  "attempt": $restart_count,
  "reason": "inventory scanner did not publish a complete result",
  "job_exit_code": $rc
}
EOF
        publish_file_no_replace "$fallback_marker_tmp" "$fallback_marker" || true
    fi
    exit "$rc"
}
trap finish EXIT

{
    printf '%q ' "$0" "${original_args[@]}"
    printf '\n'
} > "$attempt_dir/invocation.txt"
safe_slurm_variables=(
    SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST
    SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_NUM_NODES SLURM_NTASKS
    SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_RESTART_COUNT SLURMD_NODENAME
)
exec 8> "$attempt_dir/slurm_environment.txt"
for variable_name in "${safe_slurm_variables[@]}"; do
    if [[ -v "$variable_name" ]]; then
        printf '%s=%q\n' "$variable_name" "${!variable_name}" \
            >&8
    fi
done
exec 8>&-
scontrol show job -o "$job_id" > "$attempt_dir/slurm_allocation.txt" \
    2> "$attempt_dir/slurm_allocation.stderr"
[[ -s "$attempt_dir/slurm_allocation.txt" ]] || {
    printf 'Slurm allocation evidence is unavailable\n' >&2
    exit 1
}

cluster_hash=$(sha256sum "$CLUSTER_CONFIG" | awk '{print $1}')
environment_hash=$(sha256sum "$ENVIRONMENT_CONFIG" | awk '{print $1}')
job_script_hash=$(sha256sum "$JOB_SCRIPT" | awk '{print $1}')
scanner_hash=$(sha256sum "$SCANNER" | awk '{print $1}')
submitter_hash=$(sha256sum "$SUBMITTER" | awk '{print $1}')
contract_bundle_hash=$(
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
slurm_jobs_bundle_hash=$(
    find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
request_record="$CODE_ROOT/reports/slurm/submissions/requests/${DENOISENET_REQUEST_ID:?}.json"
[[ -f "$request_record" && ! -L "$request_record" ]] || {
    printf 'submission request record is missing or unsafe\n' >&2
    exit 75
}
request_hash=$(sha256sum "$request_record" | awk '{print $1}')
[[ "$cluster_hash" == "${DENOISENET_SUBMIT_CONFIG_SHA256:?}" ]] || {
    printf 'cluster configuration changed between submission and execution\n' >&2
    exit 75
}
[[ "$environment_hash" == "${DENOISENET_ENV_CONFIG_SHA256:?}" ]] || {
    printf 'environment configuration changed between submission and execution\n' >&2
    exit 75
}
[[ "$job_script_hash" == "${DENOISENET_JOB_SCRIPT_SHA256:?}" ]] || {
    printf 'job script changed between submission and execution\n' >&2
    exit 75
}
[[ "$submitter_hash" == "${DENOISENET_SUBMITTER_SHA256:?}" ]] || {
    printf 'submitter changed between submission and execution\n' >&2
    exit 75
}
[[ "$contract_bundle_hash" == "${DENOISENET_CONTRACT_BUNDLE_SHA256:?}" ]] || {
    printf 'contract helper bundle changed between submission and execution\n' >&2
    exit 75
}
[[ "$slurm_jobs_bundle_hash" == "${DENOISENET_SLURM_JOBS_BUNDLE_SHA256:?}" ]] || {
    printf 'Slurm job bundle changed between submission and execution\n' >&2
    exit 75
}
[[ "$request_hash" == "${DENOISENET_REQUEST_SHA256:?}" ]] || {
    printf 'submission request record changed between submission and execution\n' >&2
    exit 75
}

"$EEG_ENV/bin/python" "$CODE_ROOT/scripts/contract/capture_slurm_allocation.py" \
    --input "$attempt_dir/slurm_allocation.txt" \
    --output "$attempt_dir/slurm_allocation.json" \
    --expected-job-id "$job_id"

runtime_dir="$CODE_ROOT/reports/environments/eeg2025/jobs/$runtime_audit_job"
runtime_status="$runtime_dir/status.json"
runtime_probe="$runtime_dir/runtime_probe.json"
runtime_explicit="$runtime_dir/conda-explicit.txt"
runtime_explicit_hash="$runtime_dir/conda-explicit.sha256"
runtime_sanitization="$runtime_dir/conda-explicit-sanitization.json"
runtime_allocation="$runtime_dir/slurm_allocation.txt"
runtime_allocation_json="$runtime_dir/slurm_allocation.json"
runtime_request_validation="$runtime_dir/submission-request-validation.json"
runtime_pip="$runtime_dir/pip-freeze.txt"
runtime_pip_hash="$runtime_dir/pip-freeze.sha256"
runtime_pip_sanitization="$runtime_dir/pip-freeze-sanitization.json"
runtime_guard_args=()
for runtime_file in \
    "$runtime_status" "$runtime_probe" "$runtime_explicit" "$runtime_explicit_hash" \
    "$runtime_sanitization" "$runtime_allocation" "$runtime_allocation_json" \
    "$runtime_request_validation" "$runtime_pip" "$runtime_pip_hash" \
    "$runtime_pip_sanitization"; do
    if [[ ! -f "$runtime_file" || -L "$runtime_file" ]]; then
        printf 'runtime audit evidence is missing or unsafe: %s\n' "$runtime_file" >&2
        exit 1
    fi
    runtime_file_hash=$(sha256sum "$runtime_file" | awk '{print $1}')
    runtime_guard_args+=(--guard "$runtime_file" "$runtime_file_hash")
done

scanner_pid=""
scanner_wait_interrupted=false
forward_scanner_signal() {
    local signal_name=$1
    scanner_wait_interrupted=true
    if [[ -n "$scanner_pid" ]] && kill -0 "$scanner_pid" 2>/dev/null; then
        kill -s "$signal_name" "$scanner_pid" 2>/dev/null || true
    fi
}
trap 'forward_scanner_signal USR1' USR1
trap 'forward_scanner_signal TERM' TERM
trap 'forward_scanner_signal INT' INT

set +e
"$EEG_ENV/bin/python" "$SCANNER" \
    --data-root "$DATA_ROOT" \
    --output-root "$scan_dir" \
    --runtime-audit-job "$runtime_audit_job" \
    --job-id "$job_id" \
    --attempt "$restart_count" \
    --read-only \
    --guard "$CLUSTER_CONFIG" "$cluster_hash" \
    --guard "$ENVIRONMENT_CONFIG" "$environment_hash" \
    --guard "$JOB_SCRIPT" "$job_script_hash" \
    --guard "$SCANNER" "$scanner_hash" \
    --guard "$SUBMITTER" "$submitter_hash" \
    --guard "$request_record" "$request_hash" \
    "${runtime_guard_args[@]}" &
scanner_pid=$!
while true; do
    scanner_wait_interrupted=false
    wait "$scanner_pid"
    wait_rc=$?
    [[ "$scanner_wait_interrupted" == true ]] && continue
    break
done
scanner_pid=""
printf '%s\n' "$wait_rc" > "$attempt_dir/scanner_wait_exit_code.txt"
declared_exit_file="$scan_dir/process_exit_code.txt"
if [[ -f "$declared_exit_file" && ! -L "$declared_exit_file" ]]; then
    IFS= read -r declared_exit_code < "$declared_exit_file"
    [[ "$declared_exit_code" =~ ^(0|1|4|75)$ ]] || {
        printf 'scanner declared an invalid process exit code\n' >&2
        exit 1
    }
    if [[ "$wait_rc" -ne "$declared_exit_code" ]]; then
        printf 'scanner wait status disagrees with its atomic exit record\n' >&2
        exit 1
    fi
    scan_exit_code=$declared_exit_code
else
    printf 'scanner did not publish an atomic process exit record\n' >&2
    scan_exit_code=1
fi
set -e

case "$scan_exit_code" in
    0|4|75) exit "$scan_exit_code" ;;
    *)
        printf 'inventory scanner failed with exit code %s\n' "$scan_exit_code" >&2
        exit "$scan_exit_code"
        ;;
esac
