#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_mobile_headroom_v4
profile=${1:?profile required}
stage=${2:?stage required}
shift 2
dependency=""; array=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --afterok) dependency=${2:?job required}; shift 2 ;;
    --array) array=${2:?array required}; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$profile" in
  cpu) partition=CPU; cpus=2; memory=8G; walltime=02:00:00; gres="" ;;
  cpu-high) partition=cpu-high; cpus=8; memory=64G; walltime=2-00:00:00; gres="" ;;
  L40S) partition=L40S; cpus=8; memory=64G; walltime=08:00:00; gres=gpu:1 ;;
  A100) partition=A100; cpus=8; memory=64G; walltime=18:00:00; gres=gpu:1 ;;
  H100) partition=H100; cpus=15; memory=128G; walltime=18:00:00; gres=gpu:1 ;;
  *) printf 'unsupported profile: %s\n' "$profile" >&2; exit 2 ;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/mobile_bci_headroom_v4"
args=(--parsable --job-name="mbv4_${stage}" --account=c2s --qos=normal --partition="$partition"
  --cpus-per-task="$cpus" --mem="$memory" --time="$walltime"
  --output="$CODE_ROOT/slurm_logs/mobile_bci_headroom_v4/%x_%A_%a.out"
  --error="$CODE_ROOT/slurm_logs/mobile_bci_headroom_v4/%x_%A_%a.err"
  --export="ALL,DENOISENET_PROFILE=$profile")
[[ -z "$gres" ]] || args+=(--gres="$gres")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/mobile_bci_headroom_v4/job.sbatch" "$stage"

