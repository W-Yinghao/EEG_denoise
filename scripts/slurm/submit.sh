#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly CLUSTER_CONFIG="$CODE_ROOT/configs/cluster/slurm.yaml"
readonly ENVIRONMENT_CONFIG="$CODE_ROOT/configs/environments.yaml"
readonly SBATCH_BIN=/usr/bin/sbatch
umask 077
PATH=/usr/bin:/bin
export PATH
unset SLURM_CONF SLURM_CLUSTERS
while IFS='=' read -r variable_name _; do
    [[ "$variable_name" == SBATCH_* ]] && unset "$variable_name"
done < <(env)

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

cgdr_config_path() {
    local raw=${1:-configs/cgdr/klados_v4.yaml}
    local resolved
    if [[ "$raw" == /* ]]; then
        resolved=$(realpath -m -- "$raw")
    else
        resolved=$(realpath -m -- "$CODE_ROOT/$raw")
    fi
    [[ "$resolved" == "$CODE_ROOT"/* ]] || return 1
    printf '%s\n' "$resolved"
}

is_development_only_mechanism_config() {
    local config_path=$1
    [[ "${config_path##*/}" == mechanism_audit_klados_padding_repair_development.yaml ]] && return 0
    [[ -f "$config_path" && ! -L "$config_path" ]] || return 1
    awk '
        /^execution_scope:[[:space:]]*development_diagnostics_only[[:space:]]*$/ {
            found=1
        }
        END { exit(found ? 0 : 1) }
    ' "$config_path"
}

root_yaml_scalar() {
    local config_path=$1
    local field=$2
    awk -v field="$field" '
        $0 ~ "^" field ":[[:space:]]*" {
            count += 1
            line=$0
            sub("^" field ":[[:space:]]*", "", line)
            gsub(/^"|"$/, "", line)
            value=line
        }
        END {
            if (count != 1 || value == "") exit 3
            print value
        }
    ' "$config_path"
}

usage() {
    printf 'usage: %s <profile> <job> [--afterok JOB_ID] [--afternotok JOB_ID] [--array SPEC] [payload args...]\n' "$0" >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
profile=$1
job=$2
shift 2

[[ "$profile" =~ ^(cpu|cpu-high|A100|H100|L40S|V100|V100-32GB|gpu-any)$ ]] || {
    printf 'unregistered Slurm profile: %s\n' "$profile" >&2
    exit 2
}
[[ "$job" =~ ^[a-z0-9_]+$ ]] || {
    printf 'invalid job name: %s\n' "$job" >&2
    exit 2
}

job_script="$CODE_ROOT/scripts/slurm/jobs/$job.sbatch"
[[ -f "$job_script" && ! -L "$job_script" ]] || {
    printf 'unknown or unsafe job script: %s\n' "$job_script" >&2
    exit 2
}
[[ -f "$CLUSTER_CONFIG" && ! -L "$CLUSTER_CONFIG" && -r "$CLUSTER_CONFIG" ]] || {
    printf 'missing cluster configuration: %s\n' "$CLUSTER_CONFIG" >&2
    exit 2
}
root_value() {
    local field=$1
    awk -v field="$field" '
        $0 ~ "^" field ":[[:space:]]*" {
            count += 1
            line=$0
            sub("^" field ":[[:space:]]*", "", line)
            gsub(/^"|"$/, "", line)
            value=line
        }
        END {
            if (count != 1) exit 3
            print value
        }
    ' "$CLUSTER_CONFIG"
}

profile_value() {
    local field=$1
    awk -v wanted_profile="$profile" -v field="$field" '
        /^profiles:[[:space:]]*$/ { in_profiles=1; next }
        in_profiles && /^  [^[:space:]][^:]*:[[:space:]]*$/ {
            current=$0
            sub(/^  /, "", current)
            sub(/:[[:space:]]*$/, "", current)
            if (current == wanted_profile) profile_count += 1
            next
        }
        in_profiles && current == wanted_profile && $0 ~ "^    " field ":[[:space:]]*" {
            field_count += 1
            line=$0
            sub("^[[:space:]]*" field ":[[:space:]]*", "", line)
            gsub(/^"|"$/, "", line)
            value=line
        }
        END {
            if (profile_count != 1 || field_count != 1) exit 3
            print value
        }
    ' "$CLUSTER_CONFIG"
}

schema_version=$(root_value schema_version)
cluster_name=$(root_value cluster_name)
[[ "$schema_version" == "1" && "$cluster_name" == "gpucluster" ]] || {
    printf 'unsupported or unexpected cluster configuration schema/cluster\n' >&2
    exit 2
}

account=$(root_value account)
qos=$(root_value qos)
partition=$(profile_value partition)
cpus_per_task=$(profile_value cpus_per_task)
memory=$(profile_value memory)
walltime=$(profile_value walltime)
gres=$(profile_value gres)
constraint=$(profile_value constraint)
checkpoint_signal=$(profile_value checkpoint_signal)

for required in account qos partition cpus_per_task memory walltime; do
    [[ -n "${!required}" ]] || {
        printf 'cluster configuration field is empty for %s: %s\n' "$profile" "$required" >&2
        exit 2
    }
done
[[ "$account" =~ ^[A-Za-z0-9._-]+$ && "$qos" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'unsafe account or qos value in cluster configuration\n' >&2
    exit 2
}
[[ "$partition" =~ ^[A-Za-z0-9._,-]+$ && "$cpus_per_task" =~ ^[1-9][0-9]*$ ]] || {
    printf 'unsafe partition or CPU value in cluster configuration\n' >&2
    exit 2
}
[[ "$memory" =~ ^[1-9][0-9]*[KMGT]?$ ]] || {
    printf 'unsafe memory value in cluster configuration\n' >&2
    exit 2
}
[[ "$walltime" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || {
    printf 'unsafe walltime value in cluster configuration\n' >&2
    exit 2
}
[[ "$gres" == "null" || "$gres" =~ ^gpu:[1-9][0-9]*$ ]] || {
    printf 'unsafe GRES value in cluster configuration\n' >&2
    exit 2
}
readonly CONSTRAINT_PATTERN='^[A-Za-z0-9._&|-]+$'
[[ "$constraint" == "null" || "$constraint" =~ $CONSTRAINT_PATTERN ]] || {
    printf 'unsafe constraint value in cluster configuration\n' >&2
    exit 2
}
[[ "$checkpoint_signal" == "null" || "$checkpoint_signal" =~ ^[A-Z]:[A-Z0-9]+@[1-9][0-9]*$ ]] || {
    printf 'unsafe checkpoint signal value in cluster configuration\n' >&2
    exit 2
}

dependency=""
array_spec=""
extra_sbatch_args=()
payload_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok|--afternotok)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+([_][0-9]+)?(:[0-9]+([_][0-9]+)?)*$ ]] || {
                printf 'invalid dependency job ID for %s\n' "$1" >&2
                exit 2
            }
            dep_kind=${1#--}
            if [[ -n "$dependency" ]]; then
                dependency="$dependency,$dep_kind:$2"
            else
                dependency="$dep_kind:$2"
            fi
            shift 2
            ;;
        --array)
            [[ $# -ge 2 && "$2" =~ ^[0-9,%:-]+$ ]] || {
                printf 'invalid array specification\n' >&2
                exit 2
            }
            array_spec=$2
            shift 2
            ;;
        --)
            shift
            payload_args+=("$@")
            break
            ;;
        *)
            payload_args+=("$1")
            shift
            ;;
    esac
done

# The active private-project dataset workflow intentionally skips the legacy
# bundle-hash/request-JSON machinery below.  The Slurm job itself records its
# small result and terminal status under reports/.
if [[ "$job" =~ ^(dataset_harness|public_dataset_downloads|eye_bci_download|eye_bci_finalize|cgdr|cgdr_clean_replay|sgeyesub_matlab_probe|sgeyesub_reference_checkout|benchmark_source_checkout|benchmark_data_locator)$ ]]; then
    if [[ "$job" == eye_bci_download ]]; then
        [[ "$profile" == cpu-high && -n "$array_spec" ]] || {
            printf 'Eye-BCI download requires cpu-high and an array specification\n' >&2
            exit 2
        }
        if [[ ${#payload_args[@]} -eq 0 ]]; then
            [[ "$array_spec" == 0 ]] || {
                printf 'Eye-BCI pilot requires exactly --array 0\n' >&2
                exit 2
            }
        elif [[ ${#payload_args[@]} -eq 1 && "${payload_args[0]}" == remaining-shards ]]; then
            [[ "$array_spec" == '0-3%4' ]] || {
                printf 'Eye-BCI remaining download requires exactly --array 0-3%%4\n' >&2
                exit 2
            }
        else
            printf 'invalid Eye-BCI download payload\n' >&2
            exit 2
        fi
    elif [[ "$job" == cgdr ]]; then
        [[ ${#payload_args[@]} -ge 1 && ${#payload_args[@]} -le 3 ]] || {
            printf 'CGDR requires MODE [CONFIG] [STAGE]\n' >&2
            exit 2
        }
        if [[ "${payload_args[0]}" == mechanism-audit ]]; then
            stage=${payload_args[2]:-legacy-direction-check}
            case "$stage:$profile" in
                legacy-direction-check:cpu|cpu-tests:cpu|aggregate-development:cpu-high|decision:cpu-high|interpretation-audit:cpu-high|sampler-integration:L40S|sampler-integration:A100|sampler-integration:V100-32GB|sampler-integration:gpu-any|train-prior:A100|train-prior:H100|train-prior:V100-32GB|train-prior:gpu-any|development-record:A100|development-record:H100|development-record:V100-32GB|development-record:gpu-any|untouched-record:A100|untouched-record:H100|untouched-record:V100-32GB|untouched-record:gpu-any) ;;
                *) printf 'invalid mechanism-audit stage/profile combination\n' >&2; exit 2 ;;
            esac
            mechanism_config=$(cgdr_config_path "${payload_args[1]:-}") || {
                printf 'mechanism-audit config must be inside the code root\n' >&2
                exit 2
            }
            if is_development_only_mechanism_config "$mechanism_config"; then
                case "$stage" in
                    development-record|aggregate-development|untouched-record|decision|interpretation-audit)
                        printf 'padding-repair development-only config rejects stage %s\n' "$stage" >&2
                        exit 2
                        ;;
                esac
            fi
            case "$stage" in
                development-record)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'development-record requires full --array 0-7%%8 or one retry index 0-7\n' >&2; exit 2;
                    }
                    ;;
                untouched-record)
                    [[ "$array_spec" == '0-15%8' || "$array_spec" =~ ^([0-9]|1[0-5])$ ]] || {
                        printf 'untouched-record requires full --array 0-15%%8 or one retry index 0-15\n' >&2; exit 2;
                    }
                    ;;
                *)
                    [[ -z "$array_spec" ]] || {
                        printf 'this mechanism-audit stage does not accept an array\n' >&2; exit 2;
                    }
                    ;;
            esac
        elif [[ "${payload_args[0]}" == development-diagnostics ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'development-diagnostics requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                calibration-duration:cpu-high|b6-aggregate:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'development-diagnostics %s does not accept an array\n' "$stage" >&2
                        exit 2
                    }
                    ;;
                b6-record:A100|b6-record:H100|b6-record:L40S|b6-record:V100-32GB|b6-record:gpu-any)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'b6-record requires full --array 0-7%%8 or one retry index 0-7\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid development-diagnostics stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
        elif [[ "${payload_args[0]}" == sgeyesub-protocol ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'sgeyesub-protocol requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                metadata:cpu|aggregate-development:cpu-high|aggregate-evaluation:cpu-high|corrected-audit:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'sgeyesub-protocol %s does not accept an array\n' "$stage" >&2
                        exit 2
                    }
                    ;;
                development-record:cpu-high)
                    [[ "$array_spec" == '0-14%15' \
                        || "$array_spec" =~ ^([0-9]|1[0-4])$ ]] || {
                        printf 'sgeyesub development-record requires --array 0-14%%15 or one retry index 0-14\n' >&2
                        exit 2
                    }
                    ;;
                evaluation-record:cpu-high)
                    [[ "$array_spec" == '0-43%8' \
                        || "$array_spec" == '0-14%8' \
                        || "$array_spec" == '15-29%8' \
                        || "$array_spec" == '30-43%8' \
                        || "$array_spec" =~ ^([0-9]|[1-3][0-9]|4[0-3])$ ]] || {
                        printf 'sgeyesub evaluation-record requires the full 0-43%%8 array, a registered QOS shard (0-14%%8, 15-29%%8, 30-43%%8), or one retry index 0-43\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid sgeyesub-protocol stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            if [[ "$stage" == evaluation-record || "$stage" == aggregate-evaluation ]]; then
                sgeyesub_config=$(cgdr_config_path "${payload_args[1]}") || {
                    printf 'SGEYESUB config must be inside the code root\n' >&2
                    exit 2
                }
                [[ -f "$sgeyesub_config" && ! -L "$sgeyesub_config" ]] || {
                    printf 'SGEYESUB config is missing or unsafe\n' >&2
                    exit 2
                }
                development_root=$(root_yaml_scalar "$sgeyesub_config" development_output_root) || {
                    printf 'SGEYESUB config lacks one development_output_root\n' >&2
                    exit 2
                }
                if [[ "$development_root" == /* ]]; then
                    frozen_gamma="$development_root/frozen_gamma.json"
                else
                    frozen_gamma="$CODE_ROOT/$development_root/frozen_gamma.json"
                fi
                [[ -f "$frozen_gamma" && ! -L "$frozen_gamma" ]] || {
                    printf 'SGEYESUB evaluation is blocked until development gamma is frozen\n' >&2
                    exit 2
                }
            fi
        elif [[ "${payload_args[0]}" == sgeyesub-diffusion ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'sgeyesub-diffusion requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                cpu-tests:cpu)
                    [[ -z "$array_spec" ]] || {
                        printf 'sgeyesub-diffusion cpu-tests does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                integration:V100-32GB|integration:L40S|integration:A100|integration:H100|integration:gpu-any)
                    [[ -z "$array_spec" ]] || {
                        printf 'sgeyesub-diffusion integration does not accept an array\n' >&2
                        exit 2
                    }
                    [[ -z "$dependency" || "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'sgeyesub-diffusion integration accepts only an optional cpu-test afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                development-fold:V100-32GB|development-fold:L40S|development-fold:A100|development-fold:H100|development-fold:gpu-any)
                    [[ "$array_spec" == '0-9%8' || "$array_spec" =~ ^[0-9]$ ]] || {
                        printf 'sgeyesub development-fold requires --array 0-9%%8 or one retry index 0-9\n' >&2
                        exit 2
                    }
                    [[ "$array_spec" != '0-9%8' || "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'full sgeyesub development-fold array requires an afterok integration dependency\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-development:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'sgeyesub aggregate-development does not accept an array\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'sgeyesub aggregate-development requires the development array afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                evaluation-fold:V100-32GB|evaluation-fold:L40S|evaluation-fold:A100|evaluation-fold:H100|evaluation-fold:gpu-any)
                    [[ "$array_spec" == '0-14%8' || "$array_spec" =~ ^([0-9]|1[0-4])$ ]] || {
                        printf 'sgeyesub evaluation-fold requires --array 0-14%%8 or one retry index 0-14\n' >&2
                        exit 2
                    }
                    [[ "$array_spec" != '0-14%8' || "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'full sgeyesub evaluation-fold array requires the development aggregate afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-evaluation:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'sgeyesub aggregate-evaluation does not accept an array\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'sgeyesub aggregate-evaluation requires the evaluation array afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid sgeyesub-diffusion stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            sge_diffusion_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'sgeyesub-diffusion config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$sge_diffusion_config" && ! -L "$sge_diffusion_config" ]] || {
                printf 'sgeyesub-diffusion config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == stage3-deterministic ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'stage3-deterministic requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                real-record-integration:L40S|real-record-integration:A100|real-record-integration:H100|real-record-integration:V100-32GB|real-record-integration:gpu-any)
                    [[ -z "$array_spec" ]] || {
                        printf 'stage3 real-record-integration does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                train-deterministic:L40S|train-deterministic:A100|train-deterministic:H100|train-deterministic:V100-32GB|train-deterministic:gpu-any)
                    [[ "$array_spec" == '0-2%3' || "$array_spec" =~ ^[0-2]$ ]] || {
                        printf 'stage3 training requires --array 0-2%%3 or one retry index 0-2\n' >&2
                        exit 2
                    }
                    ;;
                development-record:L40S|development-record:A100|development-record:H100|development-record:V100-32GB|development-record:gpu-any)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'stage3 development-record requires --array 0-7%%8 or one retry index 0-7\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'stage3 development-record requires --afterok with the full training array job ID\n' >&2
                        exit 2
                    }
                    ;;
                historical-record:L40S|historical-record:A100|historical-record:H100|historical-record:V100-32GB|historical-record:gpu-any)
                    [[ "$array_spec" == '0-15%8' || "$array_spec" =~ ^([0-9]|1[0-5])$ ]] || {
                        printf 'stage3 historical-record requires --array 0-15%%8 or one retry index 0-15\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-development:cpu-high|aggregate-historical:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'stage3-deterministic %s does not accept an array\n' "$stage" >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid stage3-deterministic stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            stage3_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'stage3-deterministic config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$stage3_config" && ! -L "$stage3_config" ]] || {
                printf 'stage3-deterministic config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == stage3-conditional-diffusion ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'stage3-conditional-diffusion requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                train-conditional:L40S|train-conditional:A100|train-conditional:H100|train-conditional:V100-32GB|train-conditional:gpu-any)
                    [[ "$array_spec" == '0-2%3' || "$array_spec" =~ ^[0-2]$ ]] || {
                        printf 'conditional training requires --array 0-2%%3 or one retry index 0-2\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'conditional training requires --afterok with the deterministic v4 training array job ID\n' >&2
                        exit 2
                    }
                    ;;
                development-record:L40S|development-record:A100|development-record:H100|development-record:V100-32GB|development-record:gpu-any)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'conditional development-record requires --array 0-7%%8 or one retry index 0-7\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'conditional development-record requires --afterok with the full conditional training array job ID\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-development:cpu-high)
                    [[ -z "$array_spec" ]] || {
                        printf 'conditional aggregate-development does not accept an array\n' >&2
                        exit 2
                    }
                    [[ "$dependency" =~ ^afterok:[0-9]+,afterok:[0-9]+$ ]] || {
                        printf 'conditional aggregate-development requires --afterok for both conditional and deterministic full development array job IDs\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid stage3-conditional-diffusion stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            conditional_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'stage3-conditional-diffusion config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$conditional_config" && ! -L "$conditional_config" ]] || {
                printf 'stage3-conditional-diffusion config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == optimizer-step-audit ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'optimizer-step-audit requires CONFIG STAGE\n' >&2
                exit 2
            }
            [[ "${payload_args[2]}:$profile" == audit:cpu ]] || {
                printf 'optimizer-step-audit audit requires cpu\n' >&2
                exit 2
            }
            [[ -z "$array_spec" ]] || {
                printf 'optimizer-step-audit rejects arrays\n' >&2
                exit 2
            }
            optimizer_audit_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'optimizer-step-audit config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$optimizer_audit_config" && ! -L "$optimizer_audit_config" ]] || {
                printf 'optimizer-step-audit config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == eegdfus-benchmark ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'eegdfus-benchmark requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                cpu-tests:cpu)
                    [[ -z "$array_spec" ]] || {
                        printf 'eegdfus-benchmark cpu-tests does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                smoke:V100-32GB|smoke:L40S|smoke:gpu-any)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'eegdfus smoke requires --array 0-7%%8 or one retry index 0-7\n' >&2
                        exit 2
                    }
                    ;;
                full:A100|full:H100|full:gpu-any)
                    [[ "$array_spec" == '0-7%8' || "$array_spec" =~ ^[0-7]$ ]] || {
                        printf 'eegdfus full requires --array 0-7%%8 or one retry index 0-7\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-full:cpu)
                    [[ -z "$array_spec" ]] || {
                        printf 'eegdfus aggregate-full does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid eegdfus-benchmark stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            eegdfus_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'eegdfus-benchmark config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$eegdfus_config" && ! -L "$eegdfus_config" ]] || {
                printf 'eegdfus-benchmark config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == d4pm-benchmark ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'd4pm-benchmark requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                cpu-tests:cpu)
                    [[ -z "$array_spec" ]] || {
                        printf 'd4pm-benchmark cpu-tests does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                smoke:V100-32GB|smoke:L40S|smoke:gpu-any)
                    [[ "$array_spec" == '0-1%2' || "$array_spec" =~ ^[01]$ ]] || {
                        printf 'd4pm smoke requires --array 0-1%%2 or one retry index 0-1\n' >&2
                        exit 2
                    }
                    ;;
                full:A100|full:H100|full:gpu-any)
                    [[ "$array_spec" == '0-1%2' || "$array_spec" =~ ^[01]$ ]] || {
                        printf 'd4pm full requires --array 0-1%%2 or one retry index 0-1\n' >&2
                        exit 2
                    }
                    ;;
                aggregate-full:cpu)
                    [[ -z "$array_spec" ]] || {
                        printf 'd4pm aggregate-full does not accept an array\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    printf 'invalid d4pm-benchmark stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            d4pm_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'd4pm-benchmark config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$d4pm_config" && ! -L "$d4pm_config" ]] || {
                printf 'd4pm-benchmark config is missing or unsafe\n' >&2
                exit 2
            }
            # node54 silently swallows GPU jobs; the D4PM branches also need the
            # longest allocation this cluster grants before requeue.
            if [[ "$stage" == smoke || "$stage" == full ]]; then
                walltime="23:59:59"
                extra_sbatch_args+=(--exclude=node54)
            fi
        elif [[ "${payload_args[0]}" == diffusion-incremental-decision ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'diffusion-incremental-decision requires CONFIG STAGE\n' >&2
                exit 2
            }
            [[ "${payload_args[2]}:$profile" == aggregate:cpu ]] || {
                printf 'diffusion-incremental-decision aggregate requires cpu\n' >&2
                exit 2
            }
            [[ -z "$array_spec" ]] || {
                printf 'diffusion-incremental-decision rejects arrays\n' >&2
                exit 2
            }
            decision_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'diffusion-incremental-decision config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$decision_config" && ! -L "$decision_config" ]] || {
                printf 'diffusion-incremental-decision config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == diffusion-incremental-decision-v2 ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'diffusion-incremental-decision-v2 requires CONFIG STAGE\n' >&2
                exit 2
            }
            [[ "${payload_args[2]}:$profile" == aggregate:cpu ]] || {
                printf 'diffusion-incremental-decision-v2 aggregate requires cpu\n' >&2
                exit 2
            }
            [[ -z "$array_spec" ]] || {
                printf 'diffusion-incremental-decision-v2 rejects arrays\n' >&2
                exit 2
            }
            [[ "$dependency" =~ ^afterok:[0-9]+,afterok:[0-9]+$ ]] || {
                printf 'diffusion-incremental-decision-v2 requires afterok dependencies for v1 and natural SGE aggregates\n' >&2
                exit 2
            }
            decision_v2_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'diffusion-incremental-decision-v2 config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$decision_v2_config" && ! -L "$decision_v2_config" ]] || {
                printf 'diffusion-incremental-decision-v2 config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == subject-artifact-next-round ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'subject-artifact-next-round requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                a0:cpu|b1-manifest:cpu|paired-validate:cpu|b3-aggregate:cpu-high|finalize:cpu|a1:L40S|a1:A100|a1:H100|b0:L40S|b0:A100|b0:H100|b0-repair:L40S|b0-repair:A100|b0-repair:H100|b1-train:L40S|b1-train:A100|b1-train:H100|b1-worker:L40S|b1-worker:A100|b1-worker:H100|b1-paired-train:L40S|b1-paired-train:A100|b1-paired-train:H100|b2-evaluate:L40S|b2-evaluate:A100|b2-evaluate:H100|b2-worker:L40S|b2-worker:A100|b2-worker:H100|b2-paired-evaluate:L40S|b2-paired-evaluate:A100|b2-paired-evaluate:H100) ;;
                *)
                    printf 'invalid subject-artifact-next-round stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            if [[ "$stage" == b1-train || "$stage" == b2-evaluate ]]; then
                [[ "$array_spec" =~ ^0-[1-9][0-9]*%8$ ]] || {
                    printf '%s requires a manifest-derived full array with %%8 concurrency\n' "$stage" >&2
                    exit 2
                }
                [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                    printf '%s requires an afterok dependency\n' "$stage" >&2
                    exit 2
                }
            elif [[ "$stage" == b1-paired-train || "$stage" == b2-paired-evaluate ]]; then
                [[ "$array_spec" == "0-2%8" ]] || {
                    printf '%s requires the three frozen training-seed tasks with %%8 concurrency\n' "$stage" >&2
                    exit 2
                }
                [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                    printf '%s requires an afterok dependency\n' "$stage" >&2
                    exit 2
                }
            elif [[ "$stage" == b1-worker || "$stage" == b2-worker ]]; then
                [[ "$array_spec" == "0-7%8" ]] || {
                    printf '%s requires eight QoS-safe manifest worker shards\n' "$stage" >&2
                    exit 2
                }
                [[ "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                    printf '%s requires an afterok dependency\n' "$stage" >&2
                    exit 2
                }
            else
                [[ -z "$array_spec" ]] || {
                    printf 'subject-artifact-next-round %s rejects arrays\n' "$stage" >&2
                    exit 2
                }
            fi
            next_round_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'subject-artifact-next-round config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$next_round_config" && ! -L "$next_round_config" ]] || {
                printf 'subject-artifact-next-round config is missing or unsafe\n' >&2
                exit 2
            }
        elif [[ "${payload_args[0]}" == artifact-subspace-diffusion ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'artifact-subspace-diffusion requires CONFIG STAGE\n' >&2; exit 2;
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                j0:cpu|j0-eegeyenet-source:cpu|j0-eegeyenet-repository:cpu|j0-eegeyenet-wiki:cpu|j0-eegeyenet-pdf-metadata:cpu|j1-real:cpu|j8-finalize:cpu|j0-eegeyenet-download:cpu-high|j0-eegeyenet-gdrive:cpu-high|j5-aggregate:cpu-high|\
                j2-technical:L40S|j2-technical:A100|j2-technical:H100|\
                j3-train-worker:L40S|j3-train-worker:A100|j3-train-worker:H100|\
                j4-klados:L40S|j4-klados:A100|j4-klados:H100|j4-klados:gpu-any|\
                j4-sge-worker:L40S|j4-sge-worker:A100|j4-sge-worker:H100|j4-sge-worker:gpu-any) ;;
                *) printf 'invalid artifact-subspace stage/profile combination\n' >&2; exit 2 ;;
            esac
            case "$stage" in
                j3-train-worker|j4-sge-worker)
                    [[ "$array_spec" == '0-7%8' ]] || {
                        printf '%s requires --array 0-7%%8\n' "$stage" >&2; exit 2;
                    }
                    ;;
                j4-klados)
                    [[ "$array_spec" == '0-2%8' ]] || {
                        printf 'J4 Klados requires --array 0-2%%8\n' >&2; exit 2;
                    }
                    ;;
                *)
                    [[ -z "$array_spec" ]] || {
                        printf '%s rejects arrays\n' "$stage" >&2; exit 2;
                    }
                    ;;
            esac
            subspace_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'artifact-subspace config must be inside code root\n' >&2; exit 2;
            }
            [[ -f "$subspace_config" && ! -L "$subspace_config" ]] || {
                printf 'artifact-subspace config is missing or unsafe\n' >&2; exit 2;
            }
        elif [[ "${payload_args[0]}" == subject-aware-wide-v2 ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'subject-aware-wide-v2 requires CONFIG STAGE\n' >&2; exit 2;
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                j0-reaudit:cpu|j8-finalize:cpu|j1-operator-audit:cpu-high|j4-rank:cpu-high|j7-aggregate:cpu-high|\
                j0-replay:L40S|j0-replay:A100|j0-replay:H100|\
                j2-technical:L40S|j2-technical:A100|j2-technical:H100|\
                j3-carrier-worker:L40S|j3-carrier-worker:A100|j3-carrier-worker:H100|\
                j5-conditioning-worker:L40S|j5-conditioning-worker:A100|j5-conditioning-worker:H100|\
                j6-final-worker:L40S|j6-final-worker:A100|j6-final-worker:H100) ;;
                *) printf 'invalid subject-aware-wide-v2 stage/profile combination\n' >&2; exit 2 ;;
            esac
            case "$stage" in
                j3-carrier-worker|j5-conditioning-worker|j6-final-worker)
                    [[ "$array_spec" == '0-7%8' ]] || {
                        printf '%s requires --array 0-7%%8\n' "$stage" >&2; exit 2;
                    }
                    ;;
                *) [[ -z "$array_spec" ]] || { printf '%s rejects arrays\n' "$stage" >&2; exit 2; } ;;
            esac
            wide_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'wide-v2 config must be inside code root\n' >&2; exit 2;
            }
            [[ -f "$wide_config" && ! -L "$wide_config" ]] || {
                printf 'wide-v2 config is missing or unsafe\n' >&2; exit 2;
            }
        elif [[ "${payload_args[0]}" == mainline-subject-residual ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'mainline-subject-residual requires CONFIG STAGE\n' >&2; exit 2;
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                j0:cpu|j6-finalize:cpu|j5-aggregate:cpu-high|j1:L40S|j1:A100|j2-train:A100|j2-train:H100|j2-train:V100|j2-train:V100-32GB|j2-worker:A100|j2-worker:H100|j2-worker:V100|j2-worker:V100-32GB|j3-klados:L40S|j3-klados:A100|j3-klados:V100|j3-klados:V100-32GB|j3-klados:gpu-any|j4-sge:L40S|j4-sge:A100|j4-sge:H100|j4-sge:V100|j4-sge:V100-32GB|j4-sge:gpu-any|j4-worker:L40S|j4-worker:A100|j4-worker:H100|j4-worker:V100|j4-worker:V100-32GB|j4-worker:gpu-any) ;;
                *) printf 'invalid mainline subject-residual stage/profile combination\n' >&2; exit 2 ;;
            esac
            case "$stage" in
                j2-train) [[ "$array_spec" == '0-77%8' ]] || { printf 'J2 requires --array 0-77%%8\n' >&2; exit 2; } ;;
                j2-worker) [[ "$array_spec" == '0-7%8' || "$array_spec" == '5-7%8' || "$array_spec" == '7' ]] || { printf 'J2 worker requires --array 0-7%%8 or a registered recovery subset\n' >&2; exit 2; } ;;
                j4-worker) [[ "$array_spec" == '0-7%8' ]] || { printf 'J4 worker requires --array 0-7%%8\n' >&2; exit 2; } ;;
                j3-klados) [[ "$array_spec" == '0-2%8' ]] || { printf 'J3 requires --array 0-2%%8\n' >&2; exit 2; } ;;
                j4-sge) [[ "$array_spec" == '0-74%8' ]] || { printf 'J4 requires --array 0-74%%8\n' >&2; exit 2; } ;;
                *) [[ -z "$array_spec" ]] || { printf '%s rejects arrays\n' "$stage" >&2; exit 2; } ;;
            esac
            mainline_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'mainline config must be inside code root\n' >&2; exit 2;
            }
            [[ -f "$mainline_config" && ! -L "$mainline_config" ]] || {
                printf 'mainline config is missing or unsafe\n' >&2; exit 2;
            }
        elif [[ "${payload_args[0]}" == subject-artifact ]]; then
            [[ ${#payload_args[@]} -eq 3 ]] || {
                printf 'subject-artifact requires CONFIG STAGE\n' >&2
                exit 2
            }
            stage=${payload_args[2]}
            case "$stage:$profile" in
                j0-audit:cpu|j1-cpu:cpu|finalize:cpu|aggregate:cpu-high) ;;
                validity:L40S|validity:A100|validity:H100) ;;
                train:L40S|train:A100|train:H100) ;;
                evaluate:L40S|evaluate:A100|evaluate:H100) ;;
                *)
                    printf 'invalid subject-artifact stage/profile combination\n' >&2
                    exit 2
                    ;;
            esac
            case "$stage" in
                train)
                    [[ "$array_spec" == '0-149%8' \
                        || "$array_spec" =~ ^([0-9]|[1-9][0-9]|1[0-4][0-9])$ ]] || {
                        printf 'subject-artifact train requires full --array 0-149%%8 or one retry index 0-149\n' >&2
                        exit 2
                    }
                    [[ "$array_spec" != '0-149%8' \
                        || "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'full subject-artifact train array requires the validity afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                evaluate)
                    [[ "$array_spec" == '0-74%8' \
                        || "$array_spec" =~ ^([0-9]|[1-6][0-9]|7[0-4])$ ]] || {
                        printf 'subject-artifact evaluate requires full --array 0-74%%8 or one retry index 0-74\n' >&2
                        exit 2
                    }
                    [[ "$array_spec" != '0-74%8' \
                        || "$dependency" =~ ^afterok:[0-9]+$ ]] || {
                        printf 'full subject-artifact evaluate array requires the training afterok dependency\n' >&2
                        exit 2
                    }
                    ;;
                *)
                    [[ -z "$array_spec" ]] || {
                        printf 'subject-artifact %s rejects arrays\n' "$stage" >&2
                        exit 2
                    }
                    ;;
            esac
            subject_artifact_config=$(cgdr_config_path "${payload_args[1]}") || {
                printf 'subject-artifact config must be inside the code root\n' >&2
                exit 2
            }
            [[ -f "$subject_artifact_config" \
                && ! -L "$subject_artifact_config" ]] || {
                printf 'subject-artifact config is missing or unsafe\n' >&2
                exit 2
            }
        else
            [[ -z "$array_spec" && ${#payload_args[@]} -le 2 ]] || {
                printf 'legacy CGDR modes require MODE [CONFIG] and no array\n' >&2
                exit 2
            }
            case "${payload_args[0]}:$profile" in
                metadata:cpu|cpu-validate:cpu|gpu-integrate:L40S|gpu-integrate:A100|gpu-integrate:gpu-any|train-fold:L40S|train-fold:A100|train-fold:H100|train-fold:V100-32GB|train-fold:gpu-any|eye-fold:L40S|eye-fold:A100|eye-fold:H100|eye-fold:V100-32GB|eye-fold:gpu-any) ;;
                *) printf 'invalid CGDR mode/profile combination\n' >&2; exit 2 ;;
            esac
        fi
    elif [[ "$job" == cgdr_clean_replay ]]; then
        [[ "$profile" == cpu && -z "$array_spec" && ${#payload_args[@]} -eq 0 ]] || {
            printf 'CGDR clean replay requires cpu and no arguments or array\n' >&2
            exit 2
        }
    elif [[ "$job" == sgeyesub_matlab_probe || "$job" == sgeyesub_reference_checkout ]]; then
        [[ "$profile" == cpu && -z "$array_spec" && ${#payload_args[@]} -eq 0 ]] || {
            printf '%s requires cpu and no arguments or array\n' "$job" >&2
            exit 2
        }
    elif [[ "$job" == benchmark_source_checkout ]]; then
        [[ "$profile" == cpu && -z "$array_spec" && ${#payload_args[@]} -eq 1 \
            && "${payload_args[0]}" =~ ^(eegdfus|d4pm|ds-ddpm)$ ]] || {
            printf 'benchmark_source_checkout requires cpu and one of eegdfus, d4pm or ds-ddpm\n' >&2
            exit 2
        }
    elif [[ "$job" == benchmark_data_locator ]]; then
        [[ "$profile" == cpu && -z "$array_spec" && ${#payload_args[@]} -eq 0 ]] || {
            printf 'benchmark_data_locator requires cpu and no arguments or array\n' >&2
            exit 2
        }
    else
        [[ "$profile" == cpu && -z "$array_spec" ]] || {
            printf 'this lightweight dataset job requires cpu and no array\n' >&2
            exit 2
        }
    fi
    safe_ensure_code_directory "$CODE_ROOT/slurm_logs/$job" || {
        printf 'unsafe Slurm log directory ancestry\n' >&2
        exit 1
    }
    if [[ -n "$array_spec" ]]; then
        light_log_token='%x-%A_%a'
    else
        light_log_token='%x-%j'
    fi
    light_sbatch_args=(
        --parsable
        --job-name="dn-$job"
        --partition="$partition"
        --account="$account"
        --qos="$qos"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$cpus_per_task"
        --mem="$memory"
        --time="$walltime"
        --chdir="$CODE_ROOT"
        --output="$CODE_ROOT/slurm_logs/$job/$light_log_token.out"
        --error="$CODE_ROOT/slurm_logs/$job/$light_log_token.err"
        --open-mode=append
        --export="DENOISENET_PROFILE=$profile,DENOISENET_JOB=$job,PATH=/usr/bin:/bin,LANG=C.UTF-8,LC_ALL=C.UTF-8"
    )
    [[ "$gres" == null ]] || light_sbatch_args+=(--gres="$gres")
    [[ "$constraint" == null ]] || light_sbatch_args+=(--constraint="$constraint")
    [[ "$checkpoint_signal" == null ]] || light_sbatch_args+=(--signal="$checkpoint_signal")
    [[ -z "$dependency" ]] || light_sbatch_args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    [[ -z "$array_spec" ]] || light_sbatch_args+=(--array="$array_spec")
    [[ ${#extra_sbatch_args[@]} -eq 0 ]] || light_sbatch_args+=("${extra_sbatch_args[@]}")
    exec "$SBATCH_BIN" "${light_sbatch_args[@]}" "$job_script" "${payload_args[@]}"
fi

[[ -f "$ENVIRONMENT_CONFIG" && ! -L "$ENVIRONMENT_CONFIG" && -r "$ENVIRONMENT_CONFIG" ]] || {
    printf 'missing environment configuration: %s\n' "$ENVIRONMENT_CONFIG" >&2
    exit 2
}

payload_args_sha256=$(
    {
        if [[ ${#payload_args[@]} -gt 0 ]]; then
            printf '%s\0' "${payload_args[@]}"
        fi
    } | sha256sum | awk '{print $1}'
)

safe_ensure_code_directory "$CODE_ROOT/slurm_logs/$job" || {
    printf 'unsafe Slurm log directory ancestry\n' >&2
    exit 1
}
safe_ensure_code_directory "$CODE_ROOT/reports/slurm/submissions/requests" || {
    printf 'unsafe submission report directory ancestry\n' >&2
    exit 1
}
config_sha256=$(sha256sum "$CLUSTER_CONFIG" | awk '{print $1}')
environment_config_sha256=$(sha256sum "$ENVIRONMENT_CONFIG" | awk '{print $1}')
job_script_sha256=$(sha256sum "$job_script" | awk '{print $1}')
submitter_sha256=$(sha256sum "$CODE_ROOT/scripts/slurm/submit.sh" | awk '{print $1}')
contract_bundle_sha256=$(
    find "$CODE_ROOT/scripts/contract" -maxdepth 1 -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
slurm_jobs_bundle_sha256=$(
    find "$CODE_ROOT/scripts/slurm/jobs" -maxdepth 1 -type f -name '*.sbatch' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 -r sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
submitted_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
request_stamp=$(date -u +'%Y%m%dT%H%M%S%N')
request_id="$job-$request_stamp-$BASHPID"
request_tmp="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.json.partial"
request_final="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.json"
set -o noclobber
if ! exec 9> "$request_tmp"; then
    set +o noclobber
    printf 'submission request temporary path already exists or is unsafe\n' >&2
    exit 1
fi
set +o noclobber
cat >&9 <<EOF
{
  "schema_version": 1,
  "request_id": "$request_id",
  "state": "prepared_before_sbatch",
  "job": "$job",
  "profile": "$profile",
  "partition": "$partition",
  "account": "$account",
  "qos": "$qos",
  "cpus_per_task": $cpus_per_task,
  "memory": "$memory",
  "walltime": "$walltime",
  "gres": "$gres",
  "constraint": "$constraint",
  "checkpoint_signal": "$checkpoint_signal",
  "dependency": "$dependency",
  "array": "$array_spec",
  "payload_argument_count": ${#payload_args[@]},
  "payload_arguments_sha256": "$payload_args_sha256",
  "cluster_config_sha256": "$config_sha256",
  "environment_config_sha256": "$environment_config_sha256",
  "job_script_sha256": "$job_script_sha256",
  "submitter_sha256": "$submitter_sha256",
  "contract_bundle_sha256": "$contract_bundle_sha256",
  "slurm_jobs_bundle_sha256": "$slurm_jobs_bundle_sha256",
  "prepared_at_utc": "$submitted_at_utc"
}
EOF
exec 9>&-
publish_file_no_replace "$request_tmp" "$request_final" || {
    printf 'submission request publication target already exists or is unsafe\n' >&2
    exit 1
}
request_sha256=$(sha256sum "$request_final" | awk '{print $1}')

sbatch_args=(
    --parsable
    --job-name="dn-$job"
    --partition="$partition"
    --account="$account"
    --qos="$qos"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$cpus_per_task"
    --mem="$memory"
    --time="$walltime"
    --chdir="$CODE_ROOT"
    --output="$CODE_ROOT/slurm_logs/$job/%x-%j.out"
    --error="$CODE_ROOT/slurm_logs/$job/%x-%j.err"
    --open-mode=append
    --comment="denoiseNet:$request_id"
    --export="DENOISENET_PROFILE=$profile,DENOISENET_JOB=$job,DENOISENET_SUBMIT_CONFIG_SHA256=$config_sha256,DENOISENET_ENV_CONFIG_SHA256=$environment_config_sha256,DENOISENET_JOB_SCRIPT_SHA256=$job_script_sha256,DENOISENET_SUBMITTER_SHA256=$submitter_sha256,DENOISENET_CONTRACT_BUNDLE_SHA256=$contract_bundle_sha256,DENOISENET_SLURM_JOBS_BUNDLE_SHA256=$slurm_jobs_bundle_sha256,DENOISENET_PAYLOAD_ARGS_SHA256=$payload_args_sha256,DENOISENET_REQUEST_ID=$request_id,DENOISENET_REQUEST_SHA256=$request_sha256,PATH=/usr/bin:/bin,LANG=C.UTF-8,LC_ALL=C.UTF-8"
)

[[ "$gres" == "null" || -z "$gres" ]] || sbatch_args+=(--gres="$gres")
[[ "$constraint" == "null" || -z "$constraint" ]] || sbatch_args+=(--constraint="$constraint")
[[ "$checkpoint_signal" == "null" || -z "$checkpoint_signal" ]] || sbatch_args+=(--signal="$checkpoint_signal")
[[ -z "$dependency" ]] || sbatch_args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
[[ -z "$array_spec" ]] || sbatch_args+=(--array="$array_spec")
[[ ${#extra_sbatch_args[@]} -eq 0 ]] || sbatch_args+=("${extra_sbatch_args[@]}")

set +e
submission_response=$("$SBATCH_BIN" "${sbatch_args[@]}" "$job_script" "${payload_args[@]}" 2>&1)
sbatch_rc=$?
set -e
submission_digest=$(printf '%s' "$submission_response" | sha256sum | awk '{print $1}')
if [[ $sbatch_rc -ne 0 ]]; then
    failure_tmp="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.sbatch_failed.json.partial"
    failure_final="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.sbatch_failed.json"
    set -o noclobber
    if ! exec 9> "$failure_tmp"; then
        set +o noclobber
        printf 'sbatch failed and its failure record path is already occupied\n' >&2
        exit "$sbatch_rc"
    fi
    set +o noclobber
    cat >&9 <<EOF
{
  "schema_version": 1,
  "request_id": "$request_id",
  "state": "sbatch_failed",
  "sbatch_exit_code": $sbatch_rc,
  "sanitized_policy": "scheduler diagnostic was hashed in memory and was not retained",
  "scheduler_diagnostic_sha256": "$submission_digest"
}
EOF
    exec 9>&-
    publish_file_no_replace "$failure_tmp" "$failure_final" || true
    printf 'sbatch failed for request %s (diagnostic sha256 %s)\n' "$request_id" "$submission_digest" >&2
    exit "$sbatch_rc"
fi
job_id_response=""
job_id_line_count=0
while IFS= read -r response_line; do
    if [[ "$response_line" =~ ^[0-9]+(\;[A-Za-z0-9._-]+)?$ ]]; then
        job_id_response=$response_line
        job_id_line_count=$((job_id_line_count + 1))
    fi
done <<< "$submission_response"
[[ $job_id_line_count -eq 1 ]] || {
    printf 'unexpected sbatch --parsable stream (sha256 %s)\n' "$submission_digest" >&2
    exit 1
}
job_id=${job_id_response%%;*}

link_tmp="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.submitted.json.partial"
link_final="$CODE_ROOT/reports/slurm/submissions/requests/$request_id.submitted.json"
set -o noclobber
if exec 9> "$link_tmp"; then
    set +o noclobber
    if cat >&9 <<EOF
{
  "schema_version": 1,
  "request_id": "$request_id",
  "state": "submitted",
  "job_id": "$job_id",
  "request_sha256": "$request_sha256"
}
EOF
    then
        exec 9>&-
        publish_file_no_replace "$link_tmp" "$link_final" || true
    else
        exec 9>&-
    fi
else
    set +o noclobber
fi
printf '%s\n' "$job_id"
submission_tmp="$CODE_ROOT/reports/slurm/submissions/$job_id.json.partial"
submission_final="$CODE_ROOT/reports/slurm/submissions/$job_id.json"
set -o noclobber
if ! exec 9> "$submission_tmp"; then
    set +o noclobber
    printf 'warning: job %s was submitted but its post-submit temporary path is occupied; request record=%s\n' \
        "$job_id" "$request_final" >&2
    exit 0
fi
set +o noclobber
if ! cat >&9 <<EOF
{
  "schema_version": 1,
  "request_id": "$request_id",
  "job_id": "$job_id",
  "job": "$job",
  "profile": "$profile",
  "partition": "$partition",
  "account": "$account",
  "qos": "$qos",
  "cpus_per_task": $cpus_per_task,
  "memory": "$memory",
  "walltime": "$walltime",
  "gres": "$gres",
  "constraint": "$constraint",
  "checkpoint_signal": "$checkpoint_signal",
  "dependency": "$dependency",
  "array": "$array_spec",
  "cluster_config_sha256": "$config_sha256",
  "environment_config_sha256": "$environment_config_sha256",
  "job_script_sha256": "$job_script_sha256",
  "submitter_sha256": "$submitter_sha256",
  "contract_bundle_sha256": "$contract_bundle_sha256",
  "slurm_jobs_bundle_sha256": "$slurm_jobs_bundle_sha256",
  "request_sha256": "$request_sha256",
  "payload_arguments_sha256": "$payload_args_sha256",
  "submitted_at_utc": "$submitted_at_utc",
  "sbatch_response": "$job_id_response",
  "scheduler_combined_stream_sha256": "$submission_digest"
}
EOF
then
    exec 9>&-
    printf 'warning: job %s was submitted but its post-submit manifest could not be written; request record=%s\n' \
        "$job_id" "$request_final" >&2
    exit 0
fi
exec 9>&-
if ! publish_file_no_replace "$submission_tmp" "$submission_final"; then
    printf 'warning: job %s was submitted but its post-submit manifest could not be published; request record=%s\n' \
        "$job_id" "$request_final" >&2
fi
