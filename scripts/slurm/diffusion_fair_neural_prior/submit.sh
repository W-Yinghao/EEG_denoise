#!/bin/bash
set -euo pipefail
stage=$1;profile=${2:-cpu};dep=${3:-};task=${4:-};ROOT=/home/infres/yinwang/denoiseNet_diffusion_fair_neural_prior
mkdir -p "$ROOT/slurm_logs/diffusion_fair_neural_prior" "$ROOT/reports/slurm"
case "$profile" in
 cpu) p=CPU;c=4;m=32G;t=06:00:00;extra=() ;;
 high) p=cpu-high,CPU;c=12;m=96G;t=18:00:00;extra=() ;;
 gpu1) p=A100,H100,L40S;c=8;m=64G;t=12:00:00;extra=(--gres=gpu:1) ;;
 gpu9a) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=0-8%8) ;;
 gpu9b) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=9-17%8) ;;
 gpu9c) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=18-26%8) ;;
 gpu27) p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array=0-26%8) ;;
 cpu18) p=cpu-high,CPU;c=8;m=64G;t=18:00:00;extra=(--array=0-17%8) ;;
 gputask) [[ "$task" =~ ^[0-9]+$ ]]||exit 2;p=A100,H100,L40S;c=8;m=64G;t=18:00:00;extra=(--gres=gpu:1 --array="$task") ;;
 *) exit 2;;
esac
args=(--parsable --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" "${extra[@]}" --job-name="fair_$stage" --output="$ROOT/slurm_logs/diffusion_fair_neural_prior/%x_%A_%a.out" --error="$ROOT/slurm_logs/diffusion_fair_neural_prior/%x_%A_%a.err");[[ -n "$dep" ]]&&args+=(--dependency="afterok:$dep")
job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/diffusion_fair_neural_prior/job.sbatch" "$stage");meta=$dep;[[ -n "$task" ]]&&meta="${meta:+$meta;}task=$task";printf '%s|%s|%s|%s|%s\n' "$(date -u +'%FT%TZ')" "$job" "$profile" "$stage" "$meta">>"$ROOT/reports/slurm/diffusion_fair_neural_prior_job_ids.txt";echo "$job"
