#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_lit_explore_v3
readonly SBATCH=/usr/bin/sbatch
profile=${1:?profile required}
stage=${2:?stage required}
shift 2

dependency=""
dependency_mode=afterok
array=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok) dependency_mode=afterok; dependency=${2:?job ID required}; shift 2 ;;
        --afterany) dependency_mode=afterany; dependency=${2:?job ID required}; shift 2 ;;
        --array) array=${2:?array specification required}; shift 2 ;;
        *) printf 'unknown v3 submit argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$profile" in
    cpu) partition=CPU; cpus=2; memory=8G; walltime=02:00:00; gres="" ;;
    cpu-high) partition=cpu-high; cpus=8; memory=64G; walltime=5-00:00:00; gres="" ;;
    L40S) partition=L40S; cpus=8; memory=64G; walltime=04:00:00; gres=gpu:1 ;;
    A100) partition=A100; cpus=8; memory=64G; walltime=12:00:00; gres=gpu:1 ;;
    H100) partition=H100; cpus=15; memory=128G; walltime=12:00:00; gres=gpu:1 ;;
    gpu-any) partition=V100,V100-32GB,A100,A40,L40S,H100; cpus=8; memory=64G; walltime=12:00:00; gres=gpu:1 ;;
    *) printf 'unsupported v3 profile: %s\n' "$profile" >&2; exit 2 ;;
esac

mkdir -p "$CODE_ROOT/slurm_logs/literature_guided_v3"
arguments=(
    --parsable
    --job-name="litv3_${stage}"
    --account=c2s
    --qos=normal
    --partition="$partition"
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$walltime"
    --output="$CODE_ROOT/slurm_logs/literature_guided_v3/%x_%A_%a.out"
    --error="$CODE_ROOT/slurm_logs/literature_guided_v3/%x_%A_%a.err"
    --export="ALL,DENOISENET_PROFILE=$profile"
)
[[ -z "$gres" ]] || arguments+=(--gres="$gres")
[[ -z "$gres" ]] || arguments+=(--exclude=nodeaudible01)
[[ -z "$dependency" ]] || arguments+=(--dependency="$dependency_mode:$dependency")
[[ -z "$array" ]] || arguments+=(--array="$array")
exec "$SBATCH" "${arguments[@]}" \
    "$CODE_ROOT/scripts/slurm/jobs/literature_guided_v3.sbatch" \
    "$stage" configs/cgdr/literature_guided_exploration_v3/exploration.yaml
