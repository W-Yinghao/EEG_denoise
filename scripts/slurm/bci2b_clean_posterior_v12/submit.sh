#!/bin/bash
set -euo pipefail
stage=$1;profile=${2:-cpu};dep=${3:-};ROOT=/home/infres/yinwang/denoiseNet_clean_posterior_v12;mkdir -p "$ROOT/slurm_logs/bci2b_clean_posterior_v12" "$ROOT/reports/slurm"
case "$profile" in
 cpu) p=CPU;c=4;m=32G;t=06:00:00;extra=() ;;
 high) p=cpu-high,CPU;c=12;m=96G;t=18:00:00;extra=() ;;
 gpu1) p=L40S,A100,H100;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1) ;;
 screen3) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=0-2%3) ;;
 full6) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=3-8%6) ;;
 eval3) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=0-2%3) ;;
 eval6) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=3-8%6) ;;
 *) exit 2;;esac
args=(--parsable --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" "${extra[@]}" --job-name="v12_$stage" --output="$ROOT/slurm_logs/bci2b_clean_posterior_v12/%x_%A_%a.out" --error="$ROOT/slurm_logs/bci2b_clean_posterior_v12/%x_%A_%a.err");[[ -n "$dep" ]]&&args+=(--dependency="afterok:$dep");job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/bci2b_clean_posterior_v12/job.sbatch" "$stage");printf '%s|%s|%s|%s|%s\n' "$(date -u +'%FT%TZ')" "$job" "$profile" "$stage" "$dep">>"$ROOT/reports/slurm/bci2b_clean_posterior_v12_job_ids.txt";echo "$job"
