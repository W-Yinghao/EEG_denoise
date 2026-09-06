#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_calibration_transfer_v20
profile=${1:?profile required}; stage=${2:?stage required}; shift 2
dependency=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --afterok) dependency=${2:?job required}; shift 2 ;;
    --gres|--partition|A100|V100|L40S|H100|GPU|gpu) printf 'GPU/gres/partition overrides are forbidden in V20\n' >&2; exit 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$profile" in
  cpu) partition=CPU; cpus=2; memory=8G; wall=02:00:00 ;;
  cpu-high) partition=cpu-high; cpus=8; memory=64G; wall=12:00:00 ;;
  *) printf 'only cpu/cpu-high profiles are permitted\n' >&2; exit 2 ;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/calibration_transfer_v20"
args=(--parsable --job-name="v20_${stage}" --account=c2s --qos=normal --partition="$partition"
  --cpus-per-task="$cpus" --mem="$memory" --time="$wall"
  --output="$CODE_ROOT/slurm_logs/calibration_transfer_v20/%x_%j.out"
  --error="$CODE_ROOT/slurm_logs/calibration_transfer_v20/%x_%j.err")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/calibration_transfer_v20/job.sbatch" "$stage"

