#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet
readonly EEG_ENV=/home/infres/yinwang/anaconda3/envs/eeg2025
readonly GPU_ENV=/home/infres/yinwang/anaconda3/envs/icml
mode=${1:-}
config=${2:-configs/cgdr/klados_v4.yaml}
stage=${3:-}
[[ $# -le 3 ]] || { printf 'usage: cgdr.sbatch MODE [CONFIG] [STAGE]\n' >&2; exit 2; }

development_only_config=false
if [[ "${config##*/}" == mechanism_audit_klados_padding_repair_development.yaml ]]; then
    development_only_config=true
elif [[ -f "$config" ]] && awk '
    /^execution_scope:[[:space:]]*development_diagnostics_only[[:space:]]*$/ {
        found=1
    }
    END { exit(found ? 0 : 1) }
' "$config"; then
    development_only_config=true
fi

case "$mode" in
    metadata|cpu-validate)
        [[ "${DENOISENET_PROFILE:-}" == cpu ]] || { printf '%s requires cpu\n' "$mode" >&2; exit 2; }
        python_bin="$EEG_ENV/bin/python"
        ;;
    gpu-integrate)
        [[ "${DENOISENET_PROFILE:-}" == L40S \
            || "${DENOISENET_PROFILE:-}" == A100 \
            || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
            printf 'gpu-integrate requires L40S, A100 or gpu-any\n' >&2; exit 2;
        }
        python_bin="$GPU_ENV/bin/python"
        ;;
    train-fold|eye-fold)
        [[ "${DENOISENET_PROFILE:-}" == L40S \
            || "${DENOISENET_PROFILE:-}" == A100 \
            || "${DENOISENET_PROFILE:-}" == H100 \
            || "${DENOISENET_PROFILE:-}" == V100-32GB \
            || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
            printf '%s requires a registered GPU profile\n' "$mode" >&2; exit 2;
        }
        python_bin="$GPU_ENV/bin/python"
        ;;
    mechanism-audit)
        if [[ "$development_only_config" == true ]]; then
            case "$stage" in
                development-record|aggregate-development|untouched-record|decision|interpretation-audit)
                    printf 'padding-repair development-only config rejects stage %s\n' "$stage" >&2
                    exit 2
                    ;;
            esac
        fi
        case "$stage" in
            ""|legacy-direction-check|cpu-tests)
                [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
                    printf 'mechanism-audit %s requires cpu\n' "${stage:-legacy}" >&2; exit 2;
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            aggregate-development|decision|interpretation-audit)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'mechanism-audit %s requires cpu-high\n' "$stage" >&2; exit 2;
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            sampler-integration)
                [[ "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'mechanism-audit sampler-integration requires L40S, A100, V100-32GB or gpu-any\n' >&2; exit 2;
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            train-prior|development-record|untouched-record)
                [[ "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'mechanism-audit %s requires A100, H100, V100-32GB or gpu-any\n' "$stage" >&2; exit 2;
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            *) printf 'unknown mechanism-audit stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        ;;
    development-diagnostics)
        case "$stage" in
            calibration-duration)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'development-diagnostics calibration-duration requires cpu-high\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            b6-record)
                [[ "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'development-diagnostics b6-record requires a registered GPU profile\n' >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            b6-aggregate)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'development-diagnostics b6-aggregate requires cpu-high\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            *) printf 'unknown development-diagnostics stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        ;;
    sgeyesub-protocol)
        case "$stage:${DENOISENET_PROFILE:-}" in
            metadata:cpu) ;;
            development-record:cpu-high|evaluation-record:cpu-high) ;;
            aggregate-development:cpu-high|aggregate-evaluation:cpu-high|corrected-audit:cpu-high) ;;
            *)
                printf 'invalid sgeyesub-protocol stage/profile combination\n' >&2
                exit 2
                ;;
        esac
        python_bin="$EEG_ENV/bin/python"
        ;;
    stage3-deterministic)
        case "$stage" in
            real-record-integration|train-deterministic|development-record|historical-record)
                [[ "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'stage3-deterministic %s requires a registered GPU profile\n' "$stage" >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            aggregate-development|aggregate-historical)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'stage3-deterministic %s requires cpu-high\n' "$stage" >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            *) printf 'unknown stage3-deterministic stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        case "$stage" in
            train-deterministic)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$ ]] || {
                    printf 'stage3-deterministic training requires array task 0, 1 or 2\n' >&2
                    exit 2
                }
                ;;
            development-record)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$ ]] || {
                    printf 'stage3 development requires array task 0 through 7\n' >&2
                    exit 2
                }
                ;;
            historical-record)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-5])$ ]] || {
                    printf 'stage3 historical evaluation requires array task 0 through 15\n' >&2
                    exit 2
                }
                ;;
        esac
        ;;
    stage3-conditional-diffusion)
        case "$stage" in
            train-conditional|development-record)
                [[ "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'stage3-conditional-diffusion %s requires a registered GPU profile\n' "$stage" >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            aggregate-development)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'stage3-conditional-diffusion aggregate-development requires cpu-high\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            *)
                printf 'unknown stage3-conditional-diffusion stage: %s\n' "$stage" >&2
                exit 2
                ;;
        esac
        case "$stage" in
            train-conditional)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$ ]] || {
                    printf 'conditional training requires array task 0, 1 or 2\n' >&2
                    exit 2
                }
                ;;
            development-record)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$ ]] || {
                    printf 'conditional development requires array task 0 through 7\n' >&2
                    exit 2
                }
                ;;
            aggregate-development)
                [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
                    printf 'conditional aggregate-development rejects array tasks\n' >&2
                    exit 2
                }
                ;;
        esac
        ;;
    optimizer-step-audit)
        [[ "$stage" == audit ]] || {
            printf 'optimizer-step-audit requires stage audit\n' >&2
            exit 2
        }
        [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
            printf 'optimizer-step-audit audit requires cpu\n' >&2
            exit 2
        }
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
            printf 'optimizer-step-audit rejects array tasks\n' >&2
            exit 2
        }
        python_bin="$EEG_ENV/bin/python"
        ;;
    eegdfus-benchmark)
        case "$stage" in
            cpu-tests)
                [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
                    printf 'eegdfus-benchmark cpu-tests requires cpu\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            smoke)
                [[ "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'eegdfus-benchmark smoke requires V100-32GB, L40S or gpu-any\n' >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            full)
                [[ "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'eegdfus-benchmark full requires A100, H100 or gpu-any\n' >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            aggregate-full)
                [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
                    printf 'eegdfus-benchmark aggregate-full requires cpu\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            *) printf 'unknown eegdfus-benchmark stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        ;;
    sgeyesub-diffusion)
        case "$stage" in
            cpu-tests)
                [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
                    printf 'sgeyesub-diffusion cpu-tests requires cpu\n' >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            integration|development-fold|evaluation-fold)
                [[ "${DENOISENET_PROFILE:-}" == V100-32GB \
                    || "${DENOISENET_PROFILE:-}" == A100 \
                    || "${DENOISENET_PROFILE:-}" == H100 \
                    || "${DENOISENET_PROFILE:-}" == L40S \
                    || "${DENOISENET_PROFILE:-}" == gpu-any ]] || {
                    printf 'sgeyesub-diffusion %s requires a registered GPU profile\n' "$stage" >&2
                    exit 2
                }
                python_bin="$GPU_ENV/bin/python"
                ;;
            aggregate-development|aggregate-evaluation)
                [[ "${DENOISENET_PROFILE:-}" == cpu-high ]] || {
                    printf 'sgeyesub-diffusion %s requires cpu-high\n' "$stage" >&2
                    exit 2
                }
                python_bin="$EEG_ENV/bin/python"
                ;;
            *) printf 'unknown sgeyesub-diffusion stage: %s\n' "$stage" >&2; exit 2 ;;
        esac
        case "$stage" in
            development-fold)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]$ ]] || {
                    printf 'sgeyesub development-fold requires array task 0 through 9\n' >&2
                    exit 2
                }
                ;;
            evaluation-fold)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-4])$ ]] || {
                    printf 'sgeyesub evaluation-fold requires array task 0 through 14\n' >&2
                    exit 2
                }
                ;;
            *)
                [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
                    printf 'sgeyesub-diffusion %s rejects array tasks\n' "$stage" >&2
                    exit 2
                }
                ;;
        esac
        ;;
    diffusion-incremental-decision)
        [[ "$stage" == aggregate ]] || {
            printf 'diffusion-incremental-decision requires stage aggregate\n' >&2
            exit 2
        }
        [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
            printf 'diffusion-incremental-decision aggregate requires cpu\n' >&2
            exit 2
        }
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
            printf 'diffusion-incremental-decision rejects array tasks\n' >&2
            exit 2
        }
        python_bin="$EEG_ENV/bin/python"
        ;;
    diffusion-incremental-decision-v2)
        [[ "$stage" == aggregate ]] || {
            printf 'diffusion-incremental-decision-v2 requires stage aggregate\n' >&2
            exit 2
        }
        [[ "${DENOISENET_PROFILE:-}" == cpu ]] || {
            printf 'diffusion-incremental-decision-v2 aggregate requires cpu\n' >&2
            exit 2
        }
        [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
            printf 'diffusion-incremental-decision-v2 rejects array tasks\n' >&2
            exit 2
        }
        python_bin="$EEG_ENV/bin/python"
        ;;
    subject-artifact-next-round)
        case "$stage:${DENOISENET_PROFILE:-}" in
            a0:cpu|b1-manifest:cpu|b3-aggregate:cpu-high|finalize:cpu)
                python_bin="$EEG_ENV/bin/python"
                ;;
            a1:L40S|a1:A100|a1:H100|b0:L40S|b0:A100|b0:H100|\
            b0-repair:L40S|b0-repair:A100|b0-repair:H100|\
            b1-train:L40S|b1-train:A100|b1-train:H100|\
            b1-worker:L40S|b1-worker:A100|b1-worker:H100|\
            b1-paired-train:L40S|b1-paired-train:A100|b1-paired-train:H100|\
            b2-evaluate:L40S|b2-evaluate:A100|b2-evaluate:H100|\
            b2-worker:L40S|b2-worker:A100|b2-worker:H100|\
            b2-paired-evaluate:L40S|b2-paired-evaluate:A100|b2-paired-evaluate:H100)
                python_bin="$GPU_ENV/bin/python"
                ;;
            *)
                printf 'invalid subject-artifact-next-round stage/profile combination\n' >&2
                exit 2
                ;;
        esac
        if [[ "$stage" == b1-train || "$stage" == b2-evaluate ]]; then
            task_manifest="$CODE_ROOT/results/cgdr/subject_calibrated_artifact_diffusion/revisions/deterministic_calibration_screen/deterministic_task_manifest.json"
            [[ -f "$task_manifest" && ! -L "$task_manifest" ]] || {
                printf '%s requires generated deterministic task manifest\n' "$stage" >&2
                exit 2
            }
            task_count=$(awk -F: '/"task_count"[[:space:]]*:/ {gsub(/[,[:space:]]/, "", $2); print $2; exit}' "$task_manifest")
            [[ "$task_count" =~ ^[1-9][0-9]*$ \
                && "${SLURM_ARRAY_TASK_COUNT:-}" == "$task_count" \
                && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
                && "${SLURM_ARRAY_TASK_MAX:-}" == $((task_count - 1)) ]] || {
                printf '%s array shape differs from generated manifest\n' "$stage" >&2
                exit 2
            }
        elif [[ "$stage" == b1-worker || "$stage" == b2-worker ]]; then
            [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 8 \
                && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
                && "${SLURM_ARRAY_TASK_MAX:-}" == 7 ]] || {
                printf '%s array shape differs from eight worker shards\n' "$stage" >&2
                exit 2
            }
        elif [[ "$stage" == b1-paired-train || "$stage" == b2-paired-evaluate ]]; then
            [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 3 \
                && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
                && "${SLURM_ARRAY_TASK_MAX:-}" == 2 ]] || {
                printf '%s array shape differs from the three frozen seeds\n' "$stage" >&2
                exit 2
            }
        else
            [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
                printf 'subject-artifact-next-round %s rejects array tasks\n' "$stage" >&2
                exit 2
            }
        fi
        ;;
    subject-artifact)
        case "$stage:${DENOISENET_PROFILE:-}" in
            j0-audit:cpu|j1-cpu:cpu|finalize:cpu)
                python_bin="$EEG_ENV/bin/python"
                ;;
            aggregate:cpu-high)
                python_bin="$EEG_ENV/bin/python"
                ;;
            validity:L40S|validity:A100|validity:H100|\
            train:L40S|train:A100|train:H100|\
            evaluate:L40S|evaluate:A100|evaluate:H100)
                python_bin="$GPU_ENV/bin/python"
                ;;
            *)
                printf 'invalid subject-artifact stage/profile combination\n' >&2
                exit 2
                ;;
        esac
        case "$stage" in
            train)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|[1-9][0-9]|1[0-4][0-9])$ ]] || {
                    printf 'subject-artifact train requires array task 0-149\n' >&2
                    exit 2
                }
                if [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 150 ]]; then
                    [[ "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
                        && "${SLURM_ARRAY_TASK_MAX:-}" == 149 ]] || {
                        printf 'subject-artifact train full array shape is not 0-149\n' >&2
                        exit 2
                    }
                elif [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 1 ]]; then
                    [[ "${SLURM_ARRAY_TASK_MIN:-}" == "$SLURM_ARRAY_TASK_ID" \
                        && "${SLURM_ARRAY_TASK_MAX:-}" == "$SLURM_ARRAY_TASK_ID" ]] || {
                        printf 'subject-artifact train retry must contain one explicit task\n' >&2
                        exit 2
                    }
                else
                    printf 'subject-artifact train rejects partial or malformed arrays\n' >&2
                    exit 2
                fi
                ;;
            evaluate)
                [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|[1-6][0-9]|7[0-4])$ ]] || {
                    printf 'subject-artifact evaluate requires array task 0-74\n' >&2
                    exit 2
                }
                if [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 75 ]]; then
                    [[ "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
                        && "${SLURM_ARRAY_TASK_MAX:-}" == 74 ]] || {
                        printf 'subject-artifact evaluate full array shape is not 0-74\n' >&2
                        exit 2
                    }
                elif [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 1 ]]; then
                    [[ "${SLURM_ARRAY_TASK_MIN:-}" == "$SLURM_ARRAY_TASK_ID" \
                        && "${SLURM_ARRAY_TASK_MAX:-}" == "$SLURM_ARRAY_TASK_ID" ]] || {
                        printf 'subject-artifact evaluate retry must contain one explicit task\n' >&2
                        exit 2
                    }
                else
                    printf 'subject-artifact evaluate rejects partial or malformed arrays\n' >&2
                    exit 2
                fi
                ;;
            *)
                [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
                    printf 'subject-artifact %s rejects array tasks\n' "$stage" >&2
                    exit 2
                }
                ;;
        esac
        ;;
    *) printf 'unknown CGDR mode: %s\n' "$mode" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT"
export HARNESS_LEVEL=1
job_id=${SLURM_JOB_ID:?}
task_suffix=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_suffix="_${SLURM_ARRAY_TASK_ID}"
fi
stage_suffix=""
if [[ -n "$stage" ]]; then
    stage_suffix="_${stage}"
fi
run_dir="$CODE_ROOT/runs/cgdr_${mode}${stage_suffix}_${job_id}${task_suffix}"
mkdir -p "$run_dir"
cp -- "$config" "$run_dir/config.yaml"
printf '%s\n' "$job_id" > "$run_dir/slurm_job_id.txt"
git_head=$(git rev-parse --verify HEAD)
printf '%s\n' "$git_head" > "$run_dir/git_head.txt"
export DENOISENET_GIT_HEAD="$git_head"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s\n' "$SLURM_ARRAY_TASK_ID" > "$run_dir/slurm_array_task_id.txt"
fi

if [[ "$mode" == mechanism-audit && "$stage" == cpu-tests ]]; then
    "$python_bin" -m pytest -q tests/unit/test_cgdr_*.py
fi
if [[ "$mode" == eegdfus-benchmark && "$stage" == cpu-tests ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_eegdfus_benchmark.py \
        tests/unit/test_cgdr_eegdfus_routes.py \
        tests/unit/test_cgdr_diffusion_incremental_decision.py \
        tests/unit/test_cgdr_diffusion_incremental_decision_routes.py
fi
if [[ "$mode" == sgeyesub-diffusion && "$stage" == cpu-tests ]]; then
    /usr/bin/bash -n scripts/slurm/submit.sh scripts/slurm/jobs/cgdr.sbatch
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_sgeyesub_protocol.py \
        tests/unit/test_cgdr_sgeyesub_diffusion.py \
        tests/unit/test_cgdr_sgeyesub_diffusion_runner.py \
        tests/unit/test_cgdr_sgeyesub_diffusion_routes.py \
        tests/unit/test_cgdr_diffusion_incremental_decision_v2.py \
        tests/unit/test_cgdr_stage3_deterministic.py \
        tests/unit/test_cgdr_conditional_diffusion.py
    "$GPU_ENV/bin/python" -m eeg_cgdr.cli.main \
        sgeyesub-diffusion \
        --config "$config" \
        --run-dir "$run_dir/python39_compatibility" \
        --stage cpu-tests
fi
if [[ "$mode" == optimizer-step-audit && "$stage" == audit ]]; then
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_optimizer_step_audit.py \
        tests/unit/test_cgdr_optimizer_step_audit_routes.py
fi
if [[ "$mode" == subject-artifact && "$stage" == j0-audit ]]; then
    /usr/bin/bash -n scripts/slurm/submit.sh scripts/slurm/jobs/cgdr.sbatch
    "$python_bin" -m pytest -q tests/unit/test_cgdr_*.py
fi
if [[ "$mode" == subject-artifact-next-round && "$stage" == a0 ]]; then
    /usr/bin/bash -n scripts/slurm/submit.sh scripts/slurm/jobs/cgdr.sbatch
    "$python_bin" -m pytest -q \
        tests/unit/test_cgdr_latent_coordinate_semantics_r3.py \
        tests/unit/test_cgdr_artifact_latent_deterministic.py \
        tests/unit/test_cgdr_artifact_latent_diffusion.py \
        tests/unit/test_cgdr_artifact_latent_inference.py \
        tests/unit/test_cgdr_subject_artifact_training.py \
        tests/unit/test_cgdr_klados_paired_r3.py
fi
if [[ "$mode" == subject-artifact-next-round && "$stage" == finalize ]]; then
    /usr/bin/bash -n scripts/slurm/submit.sh scripts/slurm/jobs/cgdr.sbatch
    "$python_bin" -m pytest -q tests/unit/test_cgdr_*.py
fi

child_pid=""
forward_usr1() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -USR1 "$child_pid"
    fi
}
trap forward_usr1 USR1

cli_args=("$mode" --config "$config" --run-dir "$run_dir")
if [[ -n "$stage" ]]; then
    cli_args+=(--stage "$stage")
fi
"$python_bin" -m eeg_cgdr.cli.main "${cli_args[@]}" &
child_pid=$!
while true; do
    set +e
    wait "$child_pid"
    child_status=$?
    set -e
    if kill -0 "$child_pid" 2>/dev/null; then
        continue
    fi
    exit "$child_status"
done
