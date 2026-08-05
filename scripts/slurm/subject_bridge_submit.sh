#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_parallel_explore
profile=${1:?profile required}
stage=${2:?stage required}
shift 2
dependency=""; array=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok) dependency=${2:?job ID required}; shift 2 ;;
        --array) array=${2:?array required}; shift 2 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$profile" in
    cpu) partition=CPU; cpus=2; memory=8G; wall=01:00:00; gres="" ;;
    cpu-high) partition=cpu-high; cpus=8; memory=64G; wall=08:00:00; gres="" ;;
    L40S) partition=L40S; cpus=8; memory=64G; wall=04:00:00; gres=gpu:1 ;;
    A100) partition=A100; cpus=8; memory=64G; wall=18:00:00; gres=gpu:1 ;;
    H100) partition=H100; cpus=15; memory=128G; wall=18:00:00; gres=gpu:1 ;;
    A100-H100) partition=A100,H100; cpus=12; memory=96G; wall=18:00:00; gres=gpu:1 ;;
    *) printf 'unsupported profile: %s\n' "$profile" >&2; exit 2 ;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/subject_bridge_repair"
args=(--parsable --job-name="sbr_${stage}" --account=c2s --qos=normal --partition="$partition" --cpus-per-task="$cpus" --mem="$memory" --time="$wall" --output="$CODE_ROOT/slurm_logs/subject_bridge_repair/%x_%A_%a.out" --error="$CODE_ROOT/slurm_logs/subject_bridge_repair/%x_%A_%a.err" --export="ALL,DENOISENET_PROFILE=$profile")
[[ -z "$gres" ]] || args+=(--gres="$gres")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/jobs/subject_bridge.sbatch" "$stage" configs/cgdr/subject_bridge_repair.yaml
