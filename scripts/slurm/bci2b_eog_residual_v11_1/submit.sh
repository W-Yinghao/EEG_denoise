#!/bin/bash
set -euo pipefail
stage=$1;profile=${2:-cpu};dep=${3:-};ROOT=/home/infres/yinwang/denoiseNet_bci2b_eog_residual_v11_1;mkdir -p "$ROOT/slurm_logs/bci2b_eog_residual_v11_1" "$ROOT/reports/slurm"
case "$profile" in
 cpu) p=CPU;g=();c=4;m=32G;t=06:00:00;a=() ;;
 high) p=cpu-high;g=();c=12;m=96G;t=18:00:00;a=() ;;
 train6) p=L40S,A100,H100;g=(--gres=gpu:1);c=8;m=64G;t=18:00:00;a=(--array=3-8%6) ;;
 infer9) p=L40S,A100,H100;g=(--gres=gpu:1);c=8;m=64G;t=18:00:00;a=(--array=0-8%8) ;;
 eval6) p=CPU,cpu-high;g=();c=8;m=64G;t=18:00:00;a=(--array=3-8%6) ;;
 eval9) p=CPU,cpu-high;g=();c=8;m=64G;t=18:00:00;a=(--array=0-8%8) ;;
 *) exit 2;;esac
args=(--parsable --partition="$p" "${g[@]}" --cpus-per-task="$c" --mem="$m" --time="$t" "${a[@]}" --job-name="v111_$stage" --output="$ROOT/slurm_logs/bci2b_eog_residual_v11_1/%x_%A_%a.out" --error="$ROOT/slurm_logs/bci2b_eog_residual_v11_1/%x_%A_%a.err");[[ -n "$dep" ]]&&args+=(--dependency="afterok:$dep");job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/bci2b_eog_residual_v11_1/job.sbatch" "$stage");printf '%s|%s|%s|%s|%s\n' "$(date -u +'%FT%TZ')" "$job" "$profile" "$stage" "$dep">>"$ROOT/reports/slurm/bci2b_eog_residual_v11_1_job_ids.txt";echo "$job"
