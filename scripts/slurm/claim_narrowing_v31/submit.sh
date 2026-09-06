#!/usr/bin/env bash
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_claim_narrowing_v31
profile=${1:?profile required}
stage=${2:?stage required}
shift 2
array=""
throttle=8
dependency=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --array) array=$2; shift 2;;
    --throttle) throttle=$2; shift 2;;
    --dependency) [[ -n "${2:-}" ]] || { echo "empty dependency rejected" >&2; exit 2; }; dependency=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case "$profile" in
  cpu) partition=CPU; cpus=2; mem=16G; wall=04:00:00; gres=();;
  cpu-high) partition=cpu-high; cpus=8; mem=64G; wall=20:00:00; gres=();;
  gpu) partition=A100,H100,L40S,V100; cpus=8; mem=64G; wall=24:00:00; gres=(--gres=gpu:1);;
  *) echo "unknown profile: $profile" >&2; exit 2;;
esac
mkdir -p "$ROOT/slurm_logs/claim_narrowing_v31"
args=(--parsable --job-name="v31_${stage}" --account=c2s --qos=normal --partition="$partition" --cpus-per-task="$cpus" --mem="$mem" --time="$wall" "${gres[@]}" --output="$ROOT/slurm_logs/claim_narrowing_v31/%x_%A_%a.out" --error="$ROOT/slurm_logs/claim_narrowing_v31/%x_%A_%a.err")
[[ -z "$array" ]] || args+=(--array="$array%$throttle")
[[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
exec /usr/bin/sbatch "${args[@]}" "$ROOT/scripts/slurm/claim_narrowing_v31/job.sbatch" "$stage"
