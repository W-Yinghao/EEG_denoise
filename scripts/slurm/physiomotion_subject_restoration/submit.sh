#!/bin/bash
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration
mkdir -p "$ROOT/slurm_logs/physiomotion_subject_restoration" "$ROOT/reports/slurm"
case "$stage" in
 inventory) script=inventory.sbatch; args=() ;;
 download) script=download.sbatch; args=() ;;
 metadata|headroom-aggregate|tests|clean-import|archive-invalid) script=job.sbatch; args=(--partition=cpu-high,CPU --cpus-per-task=8 --mem=48G --time=12:00:00) ;;
 prepare) script=job.sbatch; args=(--partition=cpu-high,CPU --cpus-per-task=8 --mem=48G --time=18:00:00 --array=0-19%8) ;;
 headroom) script=job.sbatch; args=(--partition=cpu-high,CPU --cpus-per-task=12 --mem=96G --time=18:00:00 --array=0-4%5) ;;
 *) echo "unsupported stage $stage" >&2; exit 2 ;;
esac
dependency=${2:-}
[[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
stage_arg=()
[[ "$script" == job.sbatch ]] && stage_arg=("$stage")
job=$(sbatch --parsable "${args[@]}" --job-name="pm_${stage}" --output="$ROOT/slurm_logs/physiomotion_subject_restoration/%x_%A_%a.out" --error="$ROOT/slurm_logs/physiomotion_subject_restoration/%x_%A_%a.err" "$ROOT/scripts/slurm/physiomotion_subject_restoration/$script" "${stage_arg[@]}")
printf '%s|%s|%s\n' "$(date -u +%FT%TZ)" "$job" "$stage" >> "$ROOT/reports/slurm/physiomotion_subject_restoration_job_ids.txt"
echo "$job"
