#!/bin/bash
set -euo pipefail
stage=$1
profile=${2:-cpu}
dependency=${3:-}
ROOT=/home/infres/yinwang/denoiseNet_raw_support_clean_diffusion
mkdir -p "$ROOT/slurm_logs/raw_support_clean_diffusion" "$ROOT/reports/slurm"
case "$profile" in
 cpu) partition=CPU; cpus=4; memory=32G; wall=06:00:00; extra=() ;;
 high) partition=cpu-high,CPU; cpus=12; memory=96G; wall=18:00:00; extra=() ;;
 high9) partition=cpu-high,CPU; cpus=8; memory=64G; wall=18:00:00; extra=(--array=0-8%8) ;;
 high18) partition=cpu-high,CPU; cpus=8; memory=64G; wall=18:00:00; extra=(--array=0-17%8) ;;
 gpu1) partition=A100,H100,L40S; cpus=8; memory=64G; wall=18:00:00; extra=(--gres=gpu:1) ;;
 gpu9) partition=A100,H100,L40S; cpus=8; memory=64G; wall=18:00:00; extra=(--gres=gpu:1 --array=0-8%8) ;;
 gpu18) partition=A100,H100,L40S; cpus=8; memory=64G; wall=18:00:00; extra=(--gres=gpu:1 --array=0-17%8) ;;
 *) exit 2 ;;
esac
args=(--parsable --partition="$partition" --cpus-per-task="$cpus" --mem="$memory" --time="$wall" "${extra[@]}" --job-name="raw_${stage}" --output="$ROOT/slurm_logs/raw_support_clean_diffusion/%x_%A_%a.out" --error="$ROOT/slurm_logs/raw_support_clean_diffusion/%x_%A_%a.err")
[[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/raw_support_clean_diffusion/job.sbatch" "$stage")
printf '%s|%s|%s|%s|%s\n' "$(date -u +'%FT%TZ')" "$job" "$profile" "$stage" "$dependency" >> "$ROOT/reports/slurm/raw_support_clean_diffusion_job_ids.txt"
echo "$job"
