#!/usr/bin/env bash
set -euo pipefail
readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_scad_v22
profile=${1:?profile required};stage=${2:?stage required};shift 2
dependency="";array=""
while [[ $# -gt 0 ]];do
  case "$1" in
    --afterok) dependency=${2:?job required};shift 2;;
    --array) array=${2:?array required};shift 2;;
    *) printf 'unknown argument %s\n' "$1" >&2;exit 2;;
  esac
done
case "$profile" in
  cpu) partition=CPU;cpus=2;memory=16G;wall=04:00:00;gres=();;
  cpu-high) partition=cpu-high;cpus=8;memory=64G;wall=20:00:00;gres=();;
  gpu) partition=A100,H100,L40S;cpus=8;memory=64G;wall=24:00:00;gres=(--gres=gpu:1);;
  *) printf 'profile must be cpu, cpu-high, or gpu\n' >&2;exit 2;;
esac
mkdir -p "$CODE_ROOT/slurm_logs/scad_v22"
args=(--parsable --job-name="scadv22_${stage}" --account=c2s --qos=normal --partition="$partition" --cpus-per-task="$cpus" --mem="$memory" --time="$wall" "${gres[@]}" --output="$CODE_ROOT/slurm_logs/scad_v22/%x_%A_%a.out" --error="$CODE_ROOT/slurm_logs/scad_v22/%x_%A_%a.err")
[[ -z "$dependency" ]]||args+=(--dependency="afterok:$dependency")
[[ -z "$array" ]]||args+=(--array="$array")
exec /usr/bin/sbatch "${args[@]}" "$CODE_ROOT/scripts/slurm/scad_v22/job.sbatch" "$stage"
