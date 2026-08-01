#!/usr/bin/env bash
set -euo pipefail
umask 077
PATH=/usr/bin:/bin
export PATH

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly ENV_CONFIG="$CODE_ROOT/configs/environments.yaml"
readonly CLUSTER_CONFIG="$CODE_ROOT/configs/cluster/slurm.yaml"
readonly HELPER="$CODE_ROOT/scripts/contract/review_attachments.py"
readonly ALLOCATION_HELPER="$CODE_ROOT/scripts/contract/capture_slurm_allocation.py"
readonly SUBMITTER="$CODE_ROOT/scripts/slurm/submit.sh"
readonly REPOSITORY_JOB_SCRIPT="$CODE_ROOT/scripts/slurm/jobs/review_attachments.sbatch"
readonly MAX_ATTACHMENTS=1000
readonly MAX_INPUT_BYTES=$((256 * 1024 * 1024))
readonly MAX_TOTAL_INPUT_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_PDF_PAGES=1000
readonly MAX_PDF_PAGE_PIXELS=50000000
readonly MAX_PDF_TOTAL_PIXELS=500000000
readonly MAX_EXTRACTION_BYTES=$((2 * 1024 * 1024 * 1024))
cd "$CODE_ROOT"

original_args=("$@")
payload_count=${#original_args[@]}
payload_sha256=$(
    {
        if [[ $payload_count -gt 0 ]]; then
            printf '%s\0' "${original_args[@]}"
        fi
    } | sha256sum | awk '{print $1}'
)

runtime_audit_job=""
dependency_job=""
defer_pdf_to_registered_renderer=false
attachments=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --attachment)
            [[ $# -ge 2 ]] || { printf 'missing value for --attachment\n' >&2; exit 2; }
            attachments+=("$2")
            shift 2
            ;;
        --runtime-audit-job)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && -z "$runtime_audit_job" ]] || {
                printf 'invalid or duplicate --runtime-audit-job\n' >&2
                exit 2
            }
            runtime_audit_job=$2
            shift 2
            ;;
        --dependency-job)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && -z "$dependency_job" ]] || {
                printf 'invalid or duplicate --dependency-job\n' >&2
                exit 2
            }
            dependency_job=$2
            shift 2
            ;;
        --defer-pdf-to-registered-renderer)
            [[ "$defer_pdf_to_registered_renderer" == false ]] || {
                printf 'duplicate --defer-pdf-to-registered-renderer\n' >&2
                exit 2
            }
            defer_pdf_to_registered_renderer=true
            shift
            ;;
        *)
            printf 'unexpected attachment-review argument\n' >&2
            exit 2
            ;;
    esac
done
[[ ${#attachments[@]} -gt 0 && ${#attachments[@]} -le $MAX_ATTACHMENTS ]] || {
    printf 'attachment count is absent or exceeds the registered budget\n' >&2
    exit 2
}
[[ -n "$runtime_audit_job" ]] || { printf '--runtime-audit-job is required\n' >&2; exit 2; }
[[ -n "$dependency_job" ]] || { printf '--dependency-job is required\n' >&2; exit 2; }

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
[[ "$job_id" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'review_attachments requires one non-array Slurm job\n' >&2
    exit 2
}
[[ "${DENOISENET_PROFILE:-}" == cpu && "${DENOISENET_JOB:-}" == review_attachments && \
   "${SLURM_JOB_PARTITION:-}" == CPU && \
   "${DENOISENET_PAYLOAD_ARGS_SHA256:-}" == "$payload_sha256" && \
   "${DENOISENET_PAYLOAD_ARGS_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_REQUEST_ID:-}" =~ ^[A-Za-z0-9_.-]+$ && \
   "${DENOISENET_SUBMIT_CONFIG_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_ENV_CONFIG_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_JOB_SCRIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_SUBMITTER_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_CONTRACT_BUNDLE_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_SLURM_JOBS_BUNDLE_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${DENOISENET_REQUEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'submission provenance/profile/payload binding is absent or invalid\n' >&2
    exit 75
}
[[ -x "$EEG_ENV/bin/python" && -d "$EEG_ENV/conda-meta" ]] || {
    printf 'registered eeg2025 environment is incomplete\n' >&2
    exit 1
}
for required_file in "$HELPER" "$ALLOCATION_HELPER" "$SUBMITTER" "$REPOSITORY_JOB_SCRIPT" \
    "$ENV_CONFIG" "$CLUSTER_CONFIG"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] || {
        printf 'required attachment contract file is absent or symbolic\n' >&2
        exit 75
    }
done

sha256_of() { sha256sum -- "$1" | awk '{print $1}'; }
contract_bundle_hash() {
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}'
}
slurm_jobs_bundle_hash() {
    find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0 \
        | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}'
}

submitted_job_script_sha256=$(sha256_of "$REPOSITORY_JOB_SCRIPT")
helper_sha256=$(sha256_of "$HELPER")
environment_config_sha256=$(sha256_of "$ENV_CONFIG")
cluster_config_sha256=$(sha256_of "$CLUSTER_CONFIG")
submitter_sha256=$(sha256_of "$SUBMITTER")
contract_bundle_sha256=$(contract_bundle_hash)
slurm_jobs_bundle_sha256=$(slurm_jobs_bundle_hash)
request_record="$CODE_ROOT/reports/slurm/submissions/requests/${DENOISENET_REQUEST_ID}.json"
[[ -f "$request_record" && ! -L "$request_record" ]] || {
    printf 'submission request record is absent or unsafe\n' >&2
    exit 75
}
request_sha256=$(sha256_of "$request_record")
[[ "$submitted_job_script_sha256" == "$DENOISENET_JOB_SCRIPT_SHA256" && \
   "$environment_config_sha256" == "$DENOISENET_ENV_CONFIG_SHA256" && \
   "$cluster_config_sha256" == "$DENOISENET_SUBMIT_CONFIG_SHA256" && \
   "$submitter_sha256" == "$DENOISENET_SUBMITTER_SHA256" && \
   "$contract_bundle_sha256" == "$DENOISENET_CONTRACT_BUNDLE_SHA256" && \
   "$slurm_jobs_bundle_sha256" == "$DENOISENET_SLURM_JOBS_BUNDLE_SHA256" && \
   "$request_sha256" == "$DENOISENET_REQUEST_SHA256" ]] || {
    printf 'submission inputs changed before attachment execution\n' >&2
    exit 75
}

run_dir="$CODE_ROOT/reports/attachment_jobs/$job_id"
"$EEG_ENV/bin/python" "$HELPER" bootstrap --code-root "$CODE_ROOT" --directory "$run_dir"
started_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
status_tmp="$run_dir/status.json.partial-$job_id"
status_final="$run_dir/status.json"
published_output_root="$run_dir/outputs"
published_extraction_root="$CODE_ROOT/reports/attachment_review_extract/jobs/$job_id/review"
snapshot_root="$run_dir/snapshots"
complete_marker="$published_output_root/EXTRACTION_COMPLETE.json"

stable_inputs() {
    [[ "$(sha256_of "$REPOSITORY_JOB_SCRIPT")" == "$submitted_job_script_sha256" && \
       "$(sha256_of "$HELPER")" == "$helper_sha256" && \
       "$(sha256_of "$ENV_CONFIG")" == "$environment_config_sha256" && \
       "$(sha256_of "$CLUSTER_CONFIG")" == "$cluster_config_sha256" && \
       "$(sha256_of "$SUBMITTER")" == "$submitter_sha256" && \
       "$(contract_bundle_hash)" == "$contract_bundle_sha256" && \
       "$(slurm_jobs_bundle_hash)" == "$slurm_jobs_bundle_sha256" && \
       "$(sha256_of "$request_record")" == "$request_sha256" && \
       "${DENOISENET_PAYLOAD_ARGS_SHA256:-}" == "$payload_sha256" ]]
}

remove_complete_markers() {
    for marker in "$complete_marker"; do
        if [[ -e "$marker" || -L "$marker" ]]; then
            rm -f -- "$marker" || true
        fi
    done
}

finish() {
    rc=$?
    trap - EXIT
    if [[ $rc -eq 0 ]] && ! stable_inputs; then
        rc=75
    fi
    if [[ $rc -ne 0 ]]; then
        remove_complete_markers
    fi
    finished_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then
        if [[ "$defer_pdf_to_registered_renderer" == true ]]; then
            state=parent_attachment_phase_complete_pending_registered_pdf_renderers_and_full_read
        else
            state=extraction_complete_pending_full_read
        fi
    elif [[ $rc -eq 75 ]]; then
        state=stale_input
    else
        state=failed
    fi
    set -o noclobber
    {
        printf '{\n'
        printf '  "schema_version": 2,\n'
        printf '  "job": "review_attachments",\n'
        printf '  "job_id": "%s",\n' "$job_id"
        printf '  "runtime_audit_job_id": "%s",\n' "$runtime_audit_job"
        printf '  "dependency_job_id": "%s",\n' "$dependency_job"
        printf '  "submission_request_id": "%s",\n' "$DENOISENET_REQUEST_ID"
        printf '  "environment_name": "eeg2025",\n'
        printf '  "environment_path": "%s",\n' "$EEG_ENV"
        printf '  "profile": "cpu",\n'
        printf '  "state": "%s",\n' "$state"
        printf '  "review_complete": false,\n'
        printf '  "pdf_defer_to_registered_renderer": %s,\n' "$defer_pdf_to_registered_renderer"
        printf '  "exit_code": %s,\n' "$rc"
        printf '  "payload_arguments_sha256": "%s",\n' "$payload_sha256"
        printf '  "submitted_job_script_sha256": "%s",\n' "$submitted_job_script_sha256"
        printf '  "helper_sha256": "%s",\n' "$helper_sha256"
        printf '  "started_at_utc": "%s",\n' "$started_at_utc"
        printf '  "finished_at_utc": "%s"\n' "$finished_at_utc"
        printf '}\n'
    } > "$status_tmp" || rc=76
    set +o noclobber
    if [[ -f "$status_tmp" && ! -L "$status_tmp" ]]; then
        ln -- "$status_tmp" "$status_final" && rm -f -- "$status_tmp" || rc=76
    else
        rc=76
    fi
    exit "$rc"
}
trap finish EXIT

set -o noclobber
{
    printf 'payload_argument_count=%s\n' "$payload_count"
    printf 'payload_arguments_sha256=%s\n' "$payload_sha256"
    printf 'runtime_audit_job_id=%s\n' "$runtime_audit_job"
    printf 'dependency_job_id=%s\n' "$dependency_job"
    printf 'defer_pdf_to_registered_renderer=%s\n' "$defer_pdf_to_registered_renderer"
} > "$run_dir/invocation-binding.txt"
{
    printf 'SLURM_JOB_ID=%s\n' "$job_id"
    printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME:-}"
    printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-}"
    printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-}"
    printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-}"
    printf 'SLURM_MEM_PER_NODE=%s\n' "${SLURM_MEM_PER_NODE:-}"
    printf 'SLURM_RESTART_COUNT=%s\n' "${SLURM_RESTART_COUNT:-0}"
} > "$run_dir/slurm_environment_allowlist.txt"
set +o noclobber

allocation_raw="$run_dir/slurm_allocation.raw"
allocation_stderr="$run_dir/slurm_allocation.stderr.raw"
set -o noclobber
scontrol show job -o "$job_id" > "$allocation_raw" 2> "$allocation_stderr" || {
    set +o noclobber
    printf 'unable to capture Slurm allocation\n' >&2
    exit 1
}
set +o noclobber
"$EEG_ENV/bin/python" "$ALLOCATION_HELPER" \
    --input "$allocation_raw" \
    --output "$run_dir/slurm_allocation.json" \
    --expected-job-id "$job_id"
sha256sum "$allocation_raw" > "$run_dir/slurm_allocation_source.sha256"
rm -f -- "$allocation_raw" "$allocation_stderr"

contract_args=(
    contract
    --code-root "$CODE_ROOT"
    --job review_attachments
    --profile cpu
    --payload-sha256 "$payload_sha256"
    --payload-count "$payload_count"
    --current-job-id "$job_id"
    --runtime-audit-job "$runtime_audit_job"
    --dependency-job "$dependency_job"
    --environment-name eeg2025
    --environment-path "$EEG_ENV"
    --allocation-json "$run_dir/slurm_allocation.json"
)
"$EEG_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-start.json"

git remote -v | sha256sum > "$run_dir/git-remotes.sha256"
git status --porcelain=v2 --branch | sha256sum > "$run_dir/git-status.sha256"
if git rev-parse --verify HEAD > "$run_dir/git-commit.txt" 2> /dev/null; then
    git diff --binary HEAD | sha256sum > "$run_dir/tracked-worktree-diff.sha256"
else
    printf 'UNBORN\n' > "$run_dir/git-commit.txt"
    { git diff --binary --cached; git diff --binary; } | sha256sum > "$run_dir/tracked-worktree-diff.sha256"
fi
git ls-files --others --exclude-standard -z | sha256sum > "$run_dir/untracked-path-list.sha256"
sha256sum "$CLUSTER_CONFIG" "$ENV_CONFIG" "$REPOSITORY_JOB_SCRIPT" "$HELPER" \
    > "$run_dir/input-hashes.txt"

python_args=(
    extract
    --code-root "$CODE_ROOT"
    --output-root "$published_output_root"
    --extract-root "$published_extraction_root"
    --snapshot-root "$snapshot_root"
    --contract-validation "$run_dir/contract-validation-start.json"
    --expected-helper-sha256 "$helper_sha256"
    --max-input-bytes "$MAX_INPUT_BYTES"
    --max-total-input-bytes "$MAX_TOTAL_INPUT_BYTES"
    --max-attachments "$MAX_ATTACHMENTS"
    --max-pdf-pages "$MAX_PDF_PAGES"
    --max-pdf-page-pixels "$MAX_PDF_PAGE_PIXELS"
    --max-pdf-total-pixels "$MAX_PDF_TOTAL_PIXELS"
    --max-extraction-bytes "$MAX_EXTRACTION_BYTES"
)
for attachment in "${attachments[@]}"; do
    python_args+=(--attachment "$attachment")
done
if [[ "$defer_pdf_to_registered_renderer" == true ]]; then
    python_args+=(--defer-pdf-to-registered-renderer)
fi
PYTHONWARNINGS=error "$EEG_ENV/bin/python" "$HELPER" "${python_args[@]}"

verify_args=(
    verify
    --code-root "$CODE_ROOT"
    --output-root "$published_output_root"
    --extract-root "$published_extraction_root"
    --snapshot-root "$snapshot_root"
    --expected-helper-sha256 "$helper_sha256"
    --job-id "$job_id"
    --contract-validation "$run_dir/contract-validation-start.json"
)
"$EEG_ENV/bin/python" "$HELPER" "${verify_args[@]}" \
    --output-record "$run_dir/artifact-validation-initial.json"
"$EEG_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-postextract.json"
stable_inputs || exit 75

published_verify_args=(
    verify
    --code-root "$CODE_ROOT"
    --output-root "$published_output_root"
    --extract-root "$published_extraction_root"
    --snapshot-root "$snapshot_root"
    --expected-helper-sha256 "$helper_sha256"
    --job-id "$job_id"
    --contract-validation "$run_dir/contract-validation-start.json"
)
"$EEG_ENV/bin/python" "$HELPER" "${published_verify_args[@]}" \
    --output-record "$run_dir/artifact-validation-prefinalize.json"
"$EEG_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-prefinalize.json"
stable_inputs || exit 75
"$EEG_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-terminal.json"
stable_inputs || exit 75

"$EEG_ENV/bin/python" "$HELPER" finalize \
    --code-root "$CODE_ROOT" \
    --output-root "$published_output_root" \
    --extract-root "$published_extraction_root" \
    --snapshot-root "$snapshot_root" \
    --expected-helper-sha256 "$helper_sha256" \
    --job-id "$job_id" \
    --contract-validation "$run_dir/contract-validation-start.json" \
    --output-record "$run_dir/artifact-validation-complete.json"
"$EEG_ENV/bin/python" "$HELPER" "${published_verify_args[@]}" --require-complete \
    --output-record "$run_dir/artifact-validation-terminal.json"
stable_inputs || exit 75
sha256sum "$published_output_root/attachment_manifest.json" \
    "$published_output_root/artifacts_manifest.json" \
    "$published_output_root/READY.json" \
    "$published_output_root/EXTRACTION_COMPLETE.json" > "$run_dir/output-hashes.txt"
