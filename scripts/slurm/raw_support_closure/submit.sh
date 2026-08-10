#!/bin/bash
set -euo pipefail
stage=$1
ROOT=/home/infres/yinwang/denoiseNet_raw_support_closure
mkdir -p "$ROOT/slurm_logs/raw_support_closure" "$ROOT/reports/slurm"
job=$(sbatch --parsable --partition=cpu-high,CPU --cpus-per-task=8 --mem=32G --time=04:00:00 --job-name="closure_${stage}" --output="$ROOT/slurm_logs/raw_support_closure/%x_%j.out" --error="$ROOT/slurm_logs/raw_support_closure/%x_%j.err" "$ROOT/scripts/slurm/raw_support_closure/job.sbatch" "$stage")
printf '%s|%s|%s\n' "$(date -u +%FT%TZ)" "$job" "$stage" >> "$ROOT/reports/slurm/raw_support_closure_job_ids.txt"
echo "$job"
