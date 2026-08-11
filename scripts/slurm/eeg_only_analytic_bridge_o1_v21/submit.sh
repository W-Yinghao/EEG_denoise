#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_eeg_only_bridge_o1_v21
profile=${1:?profile required}; stage=${2:?stage required}; shift 2
dependency=""; array=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --afterok) dependency=${2:?job required}; shift 2 ;;
    --array) array=${2:?array required}; shift 2 ;;
    --gres|--partition|A100|V100|L40S|H100|GPU|gpu) printf 'GPU/gres/partition overrides are forbidden in O1-V21\n' >&2; exit 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$profile" in
  cpu) partition=CPU; cpus=2; memory=12G; wall=04:00:00 ;;
  cpu-high) partition=cpu-high; cpus=8; memory=64G; wall=16:00:00 ;;
  *) printf 'only cpu/cpu-high profiles are permitted\n' >&2; exit 2 ;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/eeg_only_analytic_bridge_o1_v21"
args=(--parsable --job-name="o1v21_${stage}" --account=c2s --qos=normal --partition="$partition"
  --cpus-per-task="$cpus" --mem="$memory" --time="$wall"
  --output="$CODE_ROOT/slurm_logs/eeg_only_analytic_bridge_o1_v21/%x_%A_%a.out"
  --error="$CODE_ROOT/slurm_logs/eeg_only_analytic_bridge_o1_v21/%x_%A_%a.err")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/eeg_only_analytic_bridge_o1_v21/job.sbatch" "$stage"

