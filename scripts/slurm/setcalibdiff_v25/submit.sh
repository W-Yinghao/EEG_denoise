#!/usr/bin/env bash
set -euo pipefail
readonly ROOT=/home/infres/yinwang/denoiseNet_setcalibdiff_v25
profile=${1:?profile};stage=${2:?stage};shift 2;array="";dependency=""
while [[ $# -gt 0 ]];do case "$1" in --array)array=$2;shift 2;;--afterok)dependency=$2;shift 2;;*)exit 2;;esac;done
case "$profile" in cpu)partition=CPU;cpus=2;mem=16G;wall=04:00:00;gres=();;cpu-high)partition=cpu-high;cpus=8;mem=64G;wall=20:00:00;gres=();;gpu)partition=A100,H100,L40S;cpus=8;mem=64G;wall=24:00:00;gres=(--gres=gpu:1);;*)exit 2;;esac
mkdir -p "$ROOT/slurm_logs/setcalibdiff_v25";args=(--parsable --job-name="v25_${stage}" --account=c2s --qos=normal --partition="$partition" --cpus-per-task="$cpus" --mem="$mem" --time="$wall" "${gres[@]}" --output="$ROOT/slurm_logs/setcalibdiff_v25/%x_%A_%a.out" --error="$ROOT/slurm_logs/setcalibdiff_v25/%x_%A_%a.err");[[ -z "$array" ]]||args+=(--array="$array");[[ -z "$dependency" ]]||args+=(--dependency="afterok:$dependency");exec /usr/bin/sbatch "${args[@]}" "$ROOT/scripts/slurm/setcalibdiff_v25/job.sbatch" "$stage"
