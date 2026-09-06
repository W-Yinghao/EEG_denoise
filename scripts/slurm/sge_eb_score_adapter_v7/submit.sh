#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_sge_transfer_v6
profile=${1:?profile};stage=${2:?stage};shift 2
after="";array="";export_vars="ALL"
while (($#));do
  case "$1" in
    --afterok) after=$2;shift 2;;
    --array) array=$2;shift 2;;
    --export) export_vars="ALL,$2";shift 2;;
    *) exit 2;;
  esac
done
case "$profile" in
  cpu) p=CPU;c=2;m=12G;t=03:00:00;g="";;
  cpu-high) p=cpu-high;c=8;m=96G;t=18:00:00;g="";;
  L40S) p=L40S;c=8;m=64G;t=08:00:00;g=gpu:1;;
  A100) p=A100;c=8;m=96G;t=18:00:00;g=gpu:1;;
  H100) p=H100;c=15;m=128G;t=18:00:00;g=gpu:1;;
  *) exit 2;;
esac
mkdir -p "$ROOT/slurm_logs/sge_eb_score_adapter_v7"
args=(--parsable --account=c2s --qos=normal --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" --job-name="v7_${stage}" --output="$ROOT/slurm_logs/sge_eb_score_adapter_v7/%x_%A_%a.out" --error="$ROOT/slurm_logs/sge_eb_score_adapter_v7/%x_%A_%a.err" --export="$export_vars")
[[ -z "$g" ]]||args+=(--gres="$g")
[[ -z "$after" ]]||args+=(--dependency="afterok:$after")
[[ -z "$array" ]]||args+=(--array="$array")
exec sbatch "${args[@]}" "$ROOT/scripts/slurm/sge_eb_score_adapter_v7/job.sbatch" "$stage"
