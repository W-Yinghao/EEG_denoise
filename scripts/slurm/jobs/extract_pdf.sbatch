#!/usr/bin/env bash
set -euo pipefail
umask 077
PATH=/usr/bin:/bin
export PATH

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly CONDA_BIN=/home/infres/yinwang/anaconda3/bin/conda
readonly ICML_ENV=/home/infres/yinwang/anaconda3/envs/icml
readonly ENV_CONFIG="$CODE_ROOT/configs/environments.yaml"
readonly CLUSTER_CONFIG="$CODE_ROOT/configs/cluster/slurm.yaml"
readonly HELPER="$CODE_ROOT/scripts/contract/extract_pdf_pymupdf.py"
readonly ALLOCATION_HELPER="$CODE_ROOT/scripts/contract/capture_slurm_allocation.py"
readonly SUBMITTER="$CODE_ROOT/scripts/slurm/submit.sh"
readonly REPOSITORY_JOB_SCRIPT="$CODE_ROOT/scripts/slurm/jobs/extract_pdf.sbatch"
readonly MAX_INPUT_BYTES=$((256 * 1024 * 1024))
readonly MAX_PDF_PAGES=1000
readonly MAX_PDF_PAGE_PIXELS=50000000
readonly MAX_PDF_TOTAL_PIXELS=500000000
readonly MAX_OUTPUT_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_RENDERER_STREAM_BYTES=$((1024 * 1024))
readonly RENDERER_STARTUP_CONTRACT_SHA256=dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae
cd "$CODE_ROOT"
ulimit -c 0

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
input=""
requested_output_dir=""
expected_sha256=""
parent_attachment_job=""
parent_manifest_sha256=""
parent_member_path=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 && -z "$input" ]] || { printf 'invalid or duplicate --input\n' >&2; exit 2; }
            input=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 && -z "$requested_output_dir" ]] || {
                printf 'invalid or duplicate --output-dir\n' >&2
                exit 2
            }
            requested_output_dir=$2
            shift 2
            ;;
        --expected-sha256)
            [[ $# -ge 2 && -z "$expected_sha256" ]] || {
                printf 'invalid or duplicate --expected-sha256\n' >&2
                exit 2
            }
            expected_sha256=$2
            shift 2
            ;;
        --parent-attachment-job)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && -z "$parent_attachment_job" ]] || {
                printf 'invalid or duplicate --parent-attachment-job\n' >&2
                exit 2
            }
            parent_attachment_job=$2
            shift 2
            ;;
        --parent-manifest-sha256)
            [[ $# -ge 2 && -z "$parent_manifest_sha256" ]] || {
                printf 'invalid or duplicate --parent-manifest-sha256\n' >&2
                exit 2
            }
            parent_manifest_sha256=$2
            shift 2
            ;;
        --parent-member-path)
            [[ $# -ge 2 && -z "$parent_member_path" && -n "$2" ]] || {
                printf 'invalid or duplicate --parent-member-path\n' >&2
                exit 2
            }
            parent_member_path=$2
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
        *)
            printf 'unexpected PDF extraction argument\n' >&2
            exit 2
            ;;
    esac
done
[[ -n "$input" && "$expected_sha256" =~ ^[0-9a-f]{64}$ && \
   "$parent_manifest_sha256" =~ ^[0-9a-f]{64}$ && \
   -n "$parent_attachment_job" && -n "$runtime_audit_job" ]] || {
    printf 'required PDF extraction binding is absent or invalid\n' >&2
    exit 2
}

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
[[ "$job_id" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'extract_pdf requires one non-array Slurm job\n' >&2
    exit 2
}
output_dir="$CODE_ROOT/reports/attachment_review_extract/jobs/$job_id"
parent_member_path_sha256=$(printf '%s' "$parent_member_path" | sha256sum | awk '{print $1}')
if [[ -n "$requested_output_dir" && "$requested_output_dir" != AUTO && \
      "$requested_output_dir" != job-scoped && "$requested_output_dir" != "$output_dir" ]]; then
    printf 'PDF output must be omitted, AUTO, job-scoped, or the exact allocated job root\n' >&2
    exit 2
fi
[[ "${DENOISENET_PROFILE:-}" == L40S && "${DENOISENET_JOB:-}" == extract_pdf && \
   "${SLURM_JOB_PARTITION:-}" == L40S && "${SLURM_GPUS_ON_NODE:-}" == 1 && \
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
[[ -x "$CONDA_BIN" && -x "$ICML_ENV/bin/python" && -d "$ICML_ENV/conda-meta" ]] || {
    printf 'registered icml environment is incomplete\n' >&2
    exit 1
}
for required_file in "$HELPER" "$ALLOCATION_HELPER" "$SUBMITTER" "$REPOSITORY_JOB_SCRIPT" \
    "$ENV_CONFIG" "$CLUSTER_CONFIG"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] || {
        printf 'required PDF contract file is absent or symbolic\n' >&2
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
    printf 'submission inputs changed before PDF execution\n' >&2
    exit 75
}

run_dir="$CODE_ROOT/reports/attachment_jobs/$job_id"
"$ICML_ENV/bin/python" "$HELPER" bootstrap --code-root "$CODE_ROOT" --directory "$run_dir"
started_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
status_tmp="$run_dir/status.json.partial-$job_id"
status_final="$run_dir/status.json"
snapshot_root="$run_dir/pdf-snapshot"
complete_marker="$output_dir/pymupdf/EXTRACTION_COMPLETE.json"
contract_validation_start_sha256=""
formal_renderer_stdout="$run_dir/formal-renderer.stdout"
formal_renderer_stderr="$run_dir/formal-renderer.stderr"
formal_renderer_stdout_fifo="$run_dir/formal-renderer.stdout.fifo"
formal_renderer_stderr_fifo="$run_dir/formal-renderer.stderr.fifo"
formal_renderer_stream_record="$run_dir/formal-renderer-stream-validation.json"
formal_renderer_stdout_sha256=""
formal_renderer_stderr_sha256=""
formal_renderer_stream_validation_sha256=""
formal_renderer_stdout_bytes=-1
formal_renderer_stderr_bytes=-1
formal_renderer_stdout_reader_pid=""
formal_renderer_stderr_reader_pid=""
formal_renderer_rc=-1
formal_renderer_stdout_reader_rc=-1
formal_renderer_stderr_reader_rc=-1
formal_renderer_streams_retained=false

renderer_stream_evidence_stable() {
    if [[ -z "$formal_renderer_stream_validation_sha256" ]]; then
        return 0
    fi
    [[ -f "$formal_renderer_stdout" && ! -L "$formal_renderer_stdout" && \
       -f "$formal_renderer_stderr" && ! -L "$formal_renderer_stderr" && \
       -f "$formal_renderer_stream_record" && ! -L "$formal_renderer_stream_record" && \
       ! -s "$formal_renderer_stdout" && ! -s "$formal_renderer_stderr" && \
       "$(sha256_of "$formal_renderer_stdout")" == "$formal_renderer_stdout_sha256" && \
       "$(sha256_of "$formal_renderer_stderr")" == "$formal_renderer_stderr_sha256" && \
       "$(sha256_of "$formal_renderer_stream_record")" == \
       "$formal_renderer_stream_validation_sha256" ]]
}

stable_inputs() {
    [[ "$(sha256_of "$REPOSITORY_JOB_SCRIPT")" == "$submitted_job_script_sha256" && \
       "$(sha256_of "$HELPER")" == "$helper_sha256" && \
       "$(sha256_of "$ENV_CONFIG")" == "$environment_config_sha256" && \
       "$(sha256_of "$CLUSTER_CONFIG")" == "$cluster_config_sha256" && \
       "$(sha256_of "$SUBMITTER")" == "$submitter_sha256" && \
       "$(contract_bundle_hash)" == "$contract_bundle_sha256" && \
       "$(slurm_jobs_bundle_hash)" == "$slurm_jobs_bundle_sha256" && \
       "$(sha256_of "$request_record")" == "$request_sha256" && \
       ( -z "$contract_validation_start_sha256" || \
         "$(sha256_of "$run_dir/contract-validation-start.json")" == \
         "$contract_validation_start_sha256" ) && \
       "${DENOISENET_PAYLOAD_ARGS_SHA256:-}" == "$payload_sha256" ]] && \
        renderer_stream_evidence_stable
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
    for reader_pid in "$formal_renderer_stdout_reader_pid" \
        "$formal_renderer_stderr_reader_pid"; do
        if [[ "$reader_pid" =~ ^[0-9]+$ ]]; then
            kill "$reader_pid" 2>/dev/null || true
            wait "$reader_pid" 2>/dev/null || true
        fi
    done
    rm -f -- "$formal_renderer_stdout_fifo" "$formal_renderer_stderr_fifo" || true
    if [[ $rc -eq 0 ]] && ! stable_inputs; then
        rc=75
    fi
    if [[ $rc -ne 0 ]]; then
        remove_complete_markers
    fi
    finished_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    if [[ $rc -eq 0 ]]; then
        state=extraction_complete_pending_full_read
    elif [[ $rc -eq 75 ]]; then
        state=stale_input
    else
        state=failed
    fi
    set -o noclobber
    {
        printf '{\n'
        printf '  "schema_version": 2,\n'
        printf '  "job": "extract_pdf",\n'
        printf '  "job_id": "%s",\n' "$job_id"
        printf '  "runtime_audit_job_id": "%s",\n' "$runtime_audit_job"
        printf '  "parent_attachment_job_id": "%s",\n' "$parent_attachment_job"
        printf '  "parent_manifest_sha256": "%s",\n' "$parent_manifest_sha256"
        printf '  "parent_member_path_sha256": "%s",\n' "$parent_member_path_sha256"
        printf '  "submission_request_id": "%s",\n' "$DENOISENET_REQUEST_ID"
        printf '  "environment_name": "icml",\n'
        printf '  "environment_path": "%s",\n' "$ICML_ENV"
        printf '  "profile": "L40S",\n'
        printf '  "state": "%s",\n' "$state"
        printf '  "review_complete": false,\n'
        printf '  "exit_code": %s,\n' "$rc"
        printf '  "expected_source_sha256": "%s",\n' "$expected_sha256"
        printf '  "payload_arguments_sha256": "%s",\n' "$payload_sha256"
        printf '  "submitted_job_script_sha256": "%s",\n' "$submitted_job_script_sha256"
        printf '  "helper_sha256": "%s",\n' "$helper_sha256"
        printf '  "formal_renderer_stream_policy": "empty",\n'
        printf '  "formal_renderer_stream_capture_bound_bytes": %s,\n' \
            "$((MAX_RENDERER_STREAM_BYTES + 1))"
        printf '  "formal_renderer_process_exit_code": %s,\n' "$formal_renderer_rc"
        printf '  "formal_renderer_stdout_reader_exit_code": %s,\n' \
            "$formal_renderer_stdout_reader_rc"
        printf '  "formal_renderer_stderr_reader_exit_code": %s,\n' \
            "$formal_renderer_stderr_reader_rc"
        printf '  "formal_renderer_streams_retained": %s,\n' \
            "$formal_renderer_streams_retained"
        printf '  "formal_renderer_stdout_bytes": %s,\n' "$formal_renderer_stdout_bytes"
        printf '  "formal_renderer_stderr_bytes": %s,\n' "$formal_renderer_stderr_bytes"
        printf '  "formal_renderer_stdout_sha256": "%s",\n' "$formal_renderer_stdout_sha256"
        printf '  "formal_renderer_stderr_sha256": "%s",\n' "$formal_renderer_stderr_sha256"
        printf '  "formal_renderer_stream_validation_sha256": "%s",\n' \
            "$formal_renderer_stream_validation_sha256"
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
    printf 'parent_attachment_job_id=%s\n' "$parent_attachment_job"
    printf 'parent_manifest_sha256=%s\n' "$parent_manifest_sha256"
    printf 'parent_member_path_sha256=%s\n' "$parent_member_path_sha256"
    printf 'expected_source_sha256=%s\n' "$expected_sha256"
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
"$ICML_ENV/bin/python" "$ALLOCATION_HELPER" \
    --input "$allocation_raw" \
    --output "$run_dir/slurm_allocation.json" \
    --expected-job-id "$job_id"
sha256sum "$allocation_raw" > "$run_dir/slurm_allocation_source.sha256"
rm -f -- "$allocation_raw" "$allocation_stderr"

contract_args=(
    contract
    --code-root "$CODE_ROOT"
    --job extract_pdf
    --profile L40S
    --payload-sha256 "$payload_sha256"
    --payload-count "$payload_count"
    --current-job-id "$job_id"
    --runtime-audit-job "$runtime_audit_job"
    --dependency-job "$parent_attachment_job"
    --environment-name icml
    --environment-path "$ICML_ENV"
    --allocation-json "$run_dir/slurm_allocation.json"
)
"$ICML_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-start.json"
contract_validation_start_sha256=$(sha256_of "$run_dir/contract-validation-start.json")
[[ "$contract_validation_start_sha256" =~ ^[0-9a-f]{64}$ ]] || exit 75
stable_inputs || exit 75
sha256sum "$CLUSTER_CONFIG" "$ENV_CONFIG" "$REPOSITORY_JOB_SCRIPT" "$HELPER" \
    > "$run_dir/input-hashes.txt"

extract_args=(
    extract
    --code-root "$CODE_ROOT"
    --input "$input"
    --output-dir "$output_dir"
    --snapshot-root "$snapshot_root"
    --contract-validation "$run_dir/contract-validation-start.json"
    --expected-contract-validation-sha256 "$contract_validation_start_sha256"
    --expected-sha256 "$expected_sha256"
    --expected-helper-sha256 "$helper_sha256"
    --parent-attachment-job "$parent_attachment_job"
    --parent-manifest-sha256 "$parent_manifest_sha256"
    --dpi 110
    --max-input-bytes "$MAX_INPUT_BYTES"
    --max-pages "$MAX_PDF_PAGES"
    --max-page-pixels "$MAX_PDF_PAGE_PIXELS"
    --max-total-pixels "$MAX_PDF_TOTAL_PIXELS"
    --max-output-bytes "$MAX_OUTPUT_BYTES"
)
if [[ -n "$parent_member_path" ]]; then
    extract_args+=("--parent-member-path=$parent_member_path")
fi
set -o noclobber
: > "$formal_renderer_stdout"
: > "$formal_renderer_stderr"
set +o noclobber
mkfifo -m 600 -- "$formal_renderer_stdout_fifo" "$formal_renderer_stderr_fifo"
head -c $((MAX_RENDERER_STREAM_BYTES + 1)) < "$formal_renderer_stdout_fifo" \
    > "$formal_renderer_stdout" &
formal_renderer_stdout_reader_pid=$!
head -c $((MAX_RENDERER_STREAM_BYTES + 1)) < "$formal_renderer_stderr_fifo" \
    > "$formal_renderer_stderr" &
formal_renderer_stderr_reader_pid=$!
set +e
"$CONDA_BIN" run --no-capture-output -p "$ICML_ENV" \
    /usr/bin/env -u PYTHONHOME -u PYTHONPATH -u PYTHONWARNINGS \
    python "$HELPER" "${extract_args[@]}" \
    > "$formal_renderer_stdout_fifo" 2> "$formal_renderer_stderr_fifo"
formal_renderer_rc=$?
wait "$formal_renderer_stdout_reader_pid"
formal_renderer_stdout_reader_rc=$?
wait "$formal_renderer_stderr_reader_pid"
formal_renderer_stderr_reader_rc=$?
formal_renderer_stdout_reader_pid=""
formal_renderer_stderr_reader_pid=""
set -e
rm -f -- "$formal_renderer_stdout_fifo" "$formal_renderer_stderr_fifo"
formal_renderer_stdout_bytes=$(wc -c < "$formal_renderer_stdout")
formal_renderer_stderr_bytes=$(wc -c < "$formal_renderer_stderr")
formal_renderer_stdout_sha256=$(sha256_of "$formal_renderer_stdout")
formal_renderer_stderr_sha256=$(sha256_of "$formal_renderer_stderr")
if [[ $formal_renderer_rc -ne 0 || $formal_renderer_stdout_reader_rc -ne 0 || \
      $formal_renderer_stderr_reader_rc -ne 0 || \
      $formal_renderer_stdout_bytes -ne 0 || \
      $formal_renderer_stderr_bytes -ne 0 ]]; then
    rm -f -- "$formal_renderer_stdout" "$formal_renderer_stderr"
    printf 'formal renderer stream contract failed (rc=%s reader_rcs=%s,%s stdout_bytes=%s stderr_bytes=%s)\n' \
        "$formal_renderer_rc" "$formal_renderer_stdout_reader_rc" \
        "$formal_renderer_stderr_reader_rc" "$formal_renderer_stdout_bytes" \
        "$formal_renderer_stderr_bytes" >&2
    if [[ $formal_renderer_rc -ne 0 ]]; then
        exit "$formal_renderer_rc"
    fi
    exit 1
fi
set -o noclobber
{
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "job_id": "%s",\n' "$job_id"
    printf '  "startup_contract_sha256": "%s",\n' \
        "$RENDERER_STARTUP_CONTRACT_SHA256"
    printf '  "policy": "empty",\n'
    printf '  "capture_bound_bytes_per_stream": %s,\n' \
        "$((MAX_RENDERER_STREAM_BYTES + 1))"
    printf '  "process_exit_code": 0,\n'
    printf '  "stdout_bytes": 0,\n'
    printf '  "stderr_bytes": 0,\n'
    printf '  "stdout_sha256": "%s",\n' "$formal_renderer_stdout_sha256"
    printf '  "stderr_sha256": "%s"\n' "$formal_renderer_stderr_sha256"
    printf '}\n'
} > "$formal_renderer_stream_record"
set +o noclobber
formal_renderer_stream_validation_sha256=$(sha256_of "$formal_renderer_stream_record")
formal_renderer_streams_retained=true
renderer_stream_evidence_stable || exit 75

verify_staging=(
    verify
    --code-root "$CODE_ROOT"
    --output-root "$output_dir"
    --expected-helper-sha256 "$helper_sha256"
    --job-id "$job_id"
    --contract-validation "$run_dir/contract-validation-start.json"
    --expected-contract-validation-sha256 "$contract_validation_start_sha256"
)
"$ICML_ENV/bin/python" "$HELPER" "${verify_staging[@]}" \
    --output-record "$run_dir/artifact-validation-initial.json"
"$ICML_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-postextract.json"
stable_inputs || exit 75
verify_published=(
    verify
    --code-root "$CODE_ROOT"
    --output-root "$output_dir"
    --expected-helper-sha256 "$helper_sha256"
    --job-id "$job_id"
    --contract-validation "$run_dir/contract-validation-start.json"
    --expected-contract-validation-sha256 "$contract_validation_start_sha256"
)
"$ICML_ENV/bin/python" "$HELPER" "${verify_published[@]}" \
    --output-record "$run_dir/artifact-validation-prefinalize.json"
"$ICML_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-prefinalize.json"
stable_inputs || exit 75
"$ICML_ENV/bin/python" "$HELPER" "${contract_args[@]}" \
    --output "$run_dir/contract-validation-terminal.json"
stable_inputs || exit 75

"$ICML_ENV/bin/python" "$HELPER" finalize \
    --code-root "$CODE_ROOT" \
    --output-root "$output_dir" \
    --expected-helper-sha256 "$helper_sha256" \
    --job-id "$job_id" \
    --contract-validation "$run_dir/contract-validation-start.json" \
    --expected-contract-validation-sha256 "$contract_validation_start_sha256" \
    --output-record "$run_dir/artifact-validation-complete.json"
"$ICML_ENV/bin/python" "$HELPER" "${verify_published[@]}" --require-complete \
    --output-record "$run_dir/artifact-validation-terminal.json"
stable_inputs || exit 75
sha256sum "$output_dir/pymupdf/document.txt" \
    "$output_dir/pymupdf/pymupdf_report.json" \
    "$output_dir/pymupdf/renderer-startup.preimport.json" \
    "$output_dir/pymupdf/renderer-startup.json" \
    "$output_dir/pymupdf/artifacts_manifest.json" \
    "$output_dir/pymupdf/READY.json" \
    "$output_dir/pymupdf/EXTRACTION_COMPLETE.json" \
    "$formal_renderer_stdout" \
    "$formal_renderer_stderr" \
    "$formal_renderer_stream_record" > "$run_dir/output-hashes.txt"
