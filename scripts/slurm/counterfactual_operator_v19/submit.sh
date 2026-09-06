#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_counterfactual_operator_v19
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
  cpu) partition=CPU; cpus=2; memory=8G; wall=04:00:00 ;;
  cpu-high) partition=cpu-high; cpus=8; memory=64G; wall=2-00:00:00 ;;
  *) printf 'v19 is CPU-only; unsupported profile: %s\n' "$profile" >&2; exit 2 ;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/counterfactual_operator_v19"
args=(--parsable --job-name="v19_${stage}" --account=c2s --qos=normal --partition="$partition"
  --cpus-per-task="$cpus" --mem="$memory" --time="$wall"
  --output="$CODE_ROOT/slurm_logs/counterfactual_operator_v19/%x_%A_%a.out"
  --error="$CODE_ROOT/slurm_logs/counterfactual_operator_v19/%x_%A_%a.err")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/counterfactual_operator_v19/job.sbatch" "$stage"
