#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly REPORT_ROOT="$CODE_ROOT/reports/sgeyesub_matlab_probe"
readonly job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}

[[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
    printf 'sgeyesub_matlab_probe requires the cpu profile\n' >&2
    exit 2
}
[[ $# -eq 0 ]] || {
    printf 'sgeyesub_matlab_probe accepts no arguments\n' >&2
    exit 2
}

cd "$CODE_ROOT"
umask 077
run_dir="$REPORT_ROOT/$job_id"
mkdir -p "$run_dir"
git rev-parse --verify HEAD > "$run_dir/git_head.txt"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"

module_command_available=false
module_matlab_listed=false
module_load_status=not_attempted
matlab_source=none
matlab_bin=""

if ! type module >/dev/null 2>&1; then
    for init_script in \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash
    do
        if [[ -r "$init_script" ]]; then
            # Site-owned module initialization; its output is suppressed.
            source "$init_script" >/dev/null 2>&1 || true
            type module >/dev/null 2>&1 && break
        fi
    done
fi
if type module >/dev/null 2>&1; then
    module_command_available=true
    set +e
    module_probe_output=$(module -t avail matlab MATLAB 2>&1)
    module_probe_rc=$?
    set -e
    if [[ $module_probe_rc -eq 0 \
        && "${module_probe_output,,}" == *matlab* ]]; then
        module_matlab_listed=true
    fi
    unset module_probe_output
fi

if command -v matlab >/dev/null 2>&1; then
    matlab_bin=$(command -v matlab)
    matlab_source=path
elif [[ "$module_command_available" == true ]]; then
    for matlab_module in matlab MATLAB; do
        set +e
        module load "$matlab_module" >/dev/null 2>&1
        module_load_rc=$?
        set -e
        if [[ $module_load_rc -eq 0 ]]; then
            module_load_status=loaded
            if command -v matlab >/dev/null 2>&1; then
                matlab_bin=$(command -v matlab)
                matlab_source=module
            else
                module_load_status=loaded_without_executable
            fi
            break
        else
            module_load_status=failed
        fi
    done
fi

matlab_command_available=false
license_test=not_run
matlab_version=unknown
if [[ -n "$matlab_bin" ]]; then
    matlab_command_available=true
    # Capture MATLAB output in memory and retain only fixed non-sensitive
    # markers.  License-server names and raw startup diagnostics are discarded.
    matlab_probe_code="try; ok=license('test','MATLAB');"
    matlab_probe_code+=" fprintf('DENOISENET_LICENSE_OK=%d\\n',ok);"
    matlab_probe_code+=" fprintf('DENOISENET_MATLAB_VERSION=%s\\n',version);"
    matlab_probe_code+=" if ~ok; exit(4); end;"
    matlab_probe_code+=" catch; fprintf('DENOISENET_PROBE_ERROR=1\\n'); exit(5); end"
    set +e
    matlab_probe_output=$(
        /usr/bin/timeout 180 "$matlab_bin" -batch \
            "$matlab_probe_code" \
            2>&1
    )
    matlab_probe_rc=$?
    set -e
    if [[ "$matlab_probe_output" == *DENOISENET_LICENSE_OK=1* \
        && $matlab_probe_rc -eq 0 ]]; then
        license_test=available
    elif [[ $matlab_probe_rc -eq 124 ]]; then
        license_test=timeout
    elif [[ "$matlab_probe_output" == *DENOISENET_LICENSE_OK=0* ]]; then
        license_test=unavailable
    else
        license_test=probe_failed
    fi
    version_line=$(printf '%s\n' "$matlab_probe_output" \
        | awk -F= '/^DENOISENET_MATLAB_VERSION=/{print $2; exit}')
    if [[ -n "$version_line" ]]; then
        matlab_version=$(printf '%s' "$version_line" \
            | tr -cd 'A-Za-z0-9 ._()-' \
            | cut -c1-80)
    fi
    unset matlab_probe_code matlab_probe_output version_line
fi

if [[ "$license_test" == available ]]; then
    parity_status=ready_for_official_code_and_fixture
elif [[ "$matlab_command_available" == false ]]; then
    parity_status=blocked_matlab_executable_unavailable
else
    parity_status=blocked_matlab_license_or_runtime_unavailable
fi

timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
result_tmp="$run_dir/availability.json.partial"
result_final="$run_dir/availability.json"
cat > "$result_tmp" <<EOF
{
  "status": "$parity_status",
  "slurm_job_id": "$job_id",
  "profile": "cpu",
  "checked_at_utc": "$timestamp",
  "command_v_matlab": "$matlab_command_available",
  "matlab_source": "$matlab_source",
  "module_command_available": "$module_command_available",
  "module_matlab_listed": "$module_matlab_listed",
  "module_load_status": "$module_load_status",
  "license_test": "$license_test",
  "matlab_version": "$matlab_version",
  "sensitive_output_policy": "raw module and license diagnostics discarded"
}
EOF
mv -- "$result_tmp" "$result_final"

printf 'status=%s\n' "$parity_status"
printf 'matlab_command_available=%s\n' "$matlab_command_available"
printf 'license_test=%s\n' "$license_test"
printf 'result=%s\n' "$result_final"
