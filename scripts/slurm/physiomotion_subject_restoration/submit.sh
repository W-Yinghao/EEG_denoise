#!/bin/bash
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration
mkdir -p "$ROOT/slurm_logs/physiomotion_subject_restoration" "$ROOT/reports/slurm"
case "$stage" in
 inventory) script=inventory.sbatch; args=() ;;
 *) echo "unsupported stage $stage" >&2; exit 2 ;;
esac
job=$(sbatch --parsable "${args[@]}" --job-name="pm_${stage}" --output="$ROOT/slurm_logs/physiomotion_subject_restoration/%x_%j.out" --error="$ROOT/slurm_logs/physiomotion_subject_restoration/%x_%j.err" "$ROOT/scripts/slurm/physiomotion_subject_restoration/$script")
printf '%s|%s|%s\n' "$(date -u +%FT%TZ)" "$job" "$stage" >> "$ROOT/reports/slurm/physiomotion_subject_restoration_job_ids.txt"
echo "$job"
