#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_basis_score_v9
profile=${1:?profile};stage=${2:?stage};shift 2;after="";array=""
while (($#));do case "$1" in --afterok) after=$2;shift 2;;--array) array=$2;shift 2;;*) exit 2;;esac;done
case "$profile" in
 cpu) p=CPU;c=2;m=16G;t=04:00:00;g="";;
 cpu-high) p=cpu-high;c=8;m=128G;t=18:00:00;g="";;
 GPU-any) p=H100,A100,L40S;c=8;m=64G;t=18:00:00;g=gpu:1;;
 *) exit 2;;
esac
mkdir -p "$ROOT/slurm_logs/sge_basis_score_factorial_v9" "$ROOT/reports/slurm"
args=(--parsable --partition="$p" --cpus-per-task="$c" --mem="$m" --time="$t" --job-name="v9_${stage}" --output="$ROOT/slurm_logs/sge_basis_score_factorial_v9/%x_%A_%a.out" --error="$ROOT/slurm_logs/sge_basis_score_factorial_v9/%x_%A_%a.err")
[[ -z "$g" ]]||args+=(--gres="$g");[[ -z "$after" ]]||args+=(--dependency="afterok:$after");[[ -z "$array" ]]||args+=(--array="$array")
job=$(sbatch "${args[@]}" "$ROOT/scripts/slurm/sge_basis_score_factorial_v9/job.sbatch" "$stage")
printf '%s|%s|%s|%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$job" "$profile" "$stage" >> "$ROOT/reports/slurm/sge_basis_score_factorial_v9_job_ids.txt";printf '%s\n' "$job"
