#!/bin/bash
set -euo pipefail
stage=$1
dependency=${2:-}
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_retrieval_fairness
mkdir -p "$ROOT/slurm_logs/physiomotion_retrieval_fairness" "$ROOT/reports/slurm"
args=(--partition=cpu-high,CPU --cpus-per-task=12 --mem=96G --time=18:00:00)
case "$stage" in
  select|evaluate) args+=(--array=0-4%5) ;;
  audit|aggregate|report|tests|clean-import) ;;
  *) echo "unsupported stage $stage" >&2; exit 2 ;;
esac
[[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
job=$(sbatch --parsable "${args[@]}" --job-name="pmj1r_${stage}" --output="$ROOT/slurm_logs/physiomotion_retrieval_fairness/%x_%A_%a.out" --error="$ROOT/slurm_logs/physiomotion_retrieval_fairness/%x_%A_%a.err" "$ROOT/scripts/slurm/physiomotion_retrieval_fairness/job.sbatch" "$stage")
printf '%s|%s|%s|%s\n' "$(date -u +%FT%TZ)" "$job" "$stage" "$dependency" >> "$ROOT/reports/slurm/physiomotion_retrieval_fairness_job_ids.txt"
echo "$job"
