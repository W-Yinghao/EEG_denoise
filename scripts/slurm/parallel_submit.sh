#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_parallel_explore
readonly SBATCH=/usr/bin/sbatch
profile=${1:?profile required}
stage=${2:?stage required}
shift 2

dependency=""
array=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok)
            dependency=${2:?dependency job ID required}; shift 2 ;;
        --array)
            array=${2:?array specification required}; shift 2 ;;
        *)
            printf 'unknown parallel-submit argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$profile" in
    cpu) partition=CPU; cpus=2; memory=8G; walltime=00:30:00; gres="" ;;
    cpu-high) partition=cpu-high; cpus=8; memory=64G; walltime=5-00:00:00; gres="" ;;
    L40S) partition=L40S; cpus=8; memory=64G; walltime=02:00:00; gres=gpu:1 ;;
    A100) partition=A100; cpus=8; memory=64G; walltime=12:00:00; gres=gpu:1 ;;
    H100) partition=H100; cpus=15; memory=128G; walltime=12:00:00; gres=gpu:1 ;;
    gpu-any) partition=V100,V100-32GB,A100,A40,L40S,H100; cpus=8; memory=64G; walltime=12:00:00; gres=gpu:1 ;;
    *) printf 'unsupported parallel profile: %s\n' "$profile" >&2; exit 2 ;;
esac

mkdir -p "$CODE_ROOT/slurm_logs/parallel_subject_routes"
arguments=(
    --parsable
    --job-name="psar_${stage}"
    --account=c2s
    --qos=normal
    --partition="$partition"
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$walltime"
    --output="$CODE_ROOT/slurm_logs/parallel_subject_routes/%x_%A_%a.out"
    --error="$CODE_ROOT/slurm_logs/parallel_subject_routes/%x_%A_%a.err"
    --export="ALL,DENOISENET_PROFILE=$profile"
)
[[ -z "$gres" ]] || arguments+=(--gres="$gres")
[[ -z "$dependency" ]] || arguments+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || arguments+=(--array="$array")
exec "$SBATCH" "${arguments[@]}" \
    "$CODE_ROOT/scripts/slurm/jobs/parallel_subject_routes.sbatch" \
    "$stage" configs/cgdr/parallel_subject_aware_routes_v1.yaml
