#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_mobile_diffusion_v5
profile=${1:?profile required};stage=${2:?stage required};shift 2;dependency="";array=""
while [[ $# -gt 0 ]]; do case "$1" in --afterok) dependency=${2:?job};shift 2;;--array) array=${2:?array};shift 2;;*) printf 'unknown: %s\n' "$1" >&2;exit 2;;esac;done
case "$profile" in
  cpu) partition=CPU;cpus=2;memory=8G;wall=02:00:00;gres="";;
  cpu-high) partition=cpu-high;cpus=8;memory=64G;wall=2-00:00:00;gres="";;
  L40S) partition=L40S;cpus=8;memory=64G;wall=12:00:00;gres=gpu:1;;
  A100) partition=A100;cpus=8;memory=64G;wall=18:00:00;gres=gpu:1;;
  H100) partition=H100;cpus=15;memory=128G;wall=18:00:00;gres=gpu:1;;
  *) printf 'unsupported profile: %s\n' "$profile" >&2;exit 2;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/mobile_temporal_v5"
args=(--parsable --job-name="mbv5_${stage}" --account=c2s --qos=normal --partition="$partition" --cpus-per-task="$cpus" --mem="$memory" --time="$wall" --output="$CODE_ROOT/slurm_logs/mobile_temporal_v5/%x_%A_%a.out" --error="$CODE_ROOT/slurm_logs/mobile_temporal_v5/%x_%A_%a.err")
[[ -z "$gres" ]]||args+=(--gres="$gres");[[ -z "$dependency" ]]||args+=(--dependency="afterok:$dependency");[[ -z "$array" ]]||args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/mobile_temporal_v5/job.sbatch" "$stage"
