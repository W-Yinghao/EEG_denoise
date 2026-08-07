#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_score_lora_v8
profile=${1:?profile}
stage=${2:?stage}
shift 2
after=""
array=""
while (($#)); do
  case "$1" in
    --afterok) after=$2; shift 2 ;;
    --array) array=$2; shift 2 ;;
    *) exit 2 ;;
  esac
done
case "$profile" in
  cpu) p=CPU; c=2; m=16G; t=04:00:00; g="" ;;
  cpu-high) p=cpu-high; c=8; m=128G; t=18:00:00; g="" ;;
  L40S) p=L40S; c=8; m=64G; t=12:00:00; g=gpu:1 ;;
  A100) p=A100; c=8; m=96G; t=18:00:00; g=gpu:1 ;;
  H100) p=H100; c=15; m=128G; t=18:00:00; g=gpu:1 ;;
  *) exit 2 ;;
esac
mkdir -p "$ROOT/slurm_logs/sge_eb_bridge_v8_1" "$ROOT/reports/slurm"
args=(--parsable --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" --job-name="v81_${stage}" --output="$ROOT/slurm_logs/sge_eb_bridge_v8_1/%x_%A_%a.out" --error="$ROOT/slurm_logs/sge_eb_bridge_v8_1/%x_%A_%a.err")
[[ -z "$g" ]] || args+=(--gres="$g")
[[ -z "$after" ]] || args+=(--dependency="afterok:$after")
[[ -z "$array" ]] || args+=(--array="$array")
job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/sge_eb_bridge_v8_1/job.sbatch" "$stage")
printf '%s|%s|%s|%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$job" "$profile" "$stage" >> "$ROOT/reports/slurm/sge_eb_bridge_v8_1_job_ids.txt"
printf '%s\n' "$job"
