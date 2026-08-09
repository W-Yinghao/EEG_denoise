#!/bin/bash
set -euo pipefail
stage=$1; profile=${2:-cpu}; dep=${3:-}; ROOT=/home/infres/yinwang/denoiseNet_subject_diffusion_explore; mkdir -p "$ROOT/slurm_logs/bci2b_subject_diffusion_mechanism_uq" "$ROOT/reports/slurm"
case "$profile" in
 cpu) p=CPU,cpu-high;c=4;m=32G;t=06:00:00;extra=() ;;
 high) p=cpu-high,CPU;c=12;m=96G;t=18:00:00;extra=() ;;
 gpu27) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=0-26%8) ;;
 eval27) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=0-26%8) ;;
 gpu9a) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=0-8%8) ;;
 gpu9b) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=9-17%8) ;;
 gpu9c) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=18-26%8) ;;
 eval9a) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=0-8%8) ;;
 eval9b) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=9-17%8) ;;
 eval9c) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=18-26%8) ;;
 *) exit 2;; esac
args=(--parsable --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" "${extra[@]}" --job-name="b_uq_$stage" --output="$ROOT/slurm_logs/bci2b_subject_diffusion_mechanism_uq/%x_%A_%a.out" --error="$ROOT/slurm_logs/bci2b_subject_diffusion_mechanism_uq/%x_%A_%a.err"); [[ -n "$dep" ]]&&args+=(--dependency="afterok:$dep"); job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/bci2b_subject_diffusion_mechanism_uq/job.sbatch" "$stage"); printf '%s|%s|%s|%s|%s\n' "$(date -u +'%FT%TZ')" "$job" "$profile" "$stage" "$dep">>"$ROOT/reports/slurm/bci2b_subject_diffusion_mechanism_uq_job_ids.txt"; echo "$job"
