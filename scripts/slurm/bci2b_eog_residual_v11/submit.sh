#!/bin/bash
set -euo pipefail
stage=$1;profile=${2:-cpu};dependency=${3:-}
ROOT=/home/infres/yinwang/denoiseNet_bci2b_eog_residual_v11
mkdir -p "$ROOT/slurm_logs/bci2b_eog_residual_v11" "$ROOT/reports/slurm"
case "$profile" in
 cpu) partition=CPU;gres=();cpus=4;mem=24G;time=04:00:00;array=() ;;
 cpu-high) partition=cpu-high;gres=();cpus=12;mem=96G;time=12:00:00;array=() ;;
 audit-array) partition=cpu-high;gres=();cpus=4;mem=24G;time=12:00:00;array=(--array=0-8%8) ;;
 cpu-array3) partition=CPU,cpu-high;gres=();cpus=6;mem=48G;time=12:00:00;array=(--array=0-2%3) ;;
 cpu-array6) partition=CPU,cpu-high;gres=();cpus=6;mem=48G;time=12:00:00;array=(--array=3-8%6) ;;
 l40s) partition=L40S,A100,H100;gres=(--gres=gpu:1);cpus=8;mem=64G;time=12:00:00;array=() ;;
 gpu-array3) partition=L40S,A100,H100;gres=(--gres=gpu:1);cpus=8;mem=64G;time=18:00:00;array=(--array=0-2%3) ;;
 gpu-array6) partition=L40S,A100,H100;gres=(--gres=gpu:1);cpus=8;mem=64G;time=18:00:00;array=(--array=3-8%6) ;;
 gpu-array9) partition=L40S,A100,H100;gres=(--gres=gpu:1);cpus=8;mem=64G;time=18:00:00;array=(--array=0-8%8) ;;
 *) echo "unknown profile $profile" >&2; exit 2 ;;
esac
args=(--parsable --partition="$partition" "${gres[@]}" --cpus-per-task="$cpus" --mem="$mem" --time="$time" "${array[@]}" --job-name="v11_${stage}" --output="$ROOT/slurm_logs/bci2b_eog_residual_v11/%x_%A_%a.out" --error="$ROOT/slurm_logs/bci2b_eog_residual_v11/%x_%A_%a.err")
[[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/bci2b_eog_residual_v11/job.sbatch" "$stage")
printf '%s|%s|%s|%s|%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$job" "$profile" "$stage" "$dependency" >> "$ROOT/reports/slurm/bci2b_eog_residual_v11_job_ids.txt"
printf '%s\n' "$job"
