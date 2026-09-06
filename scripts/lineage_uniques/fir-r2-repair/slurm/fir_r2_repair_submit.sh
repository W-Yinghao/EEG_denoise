#!/usr/bin/env bash
set -euo pipefail

readonly CODE_ROOT=/home/infres/yinwang/denoiseNet_fir_r2_repair
profile=${1:?profile is required}
stage=${2:?stage is required}
shift 2

dependency=""
array=""
exclude=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --afterok) dependency=${2:?dependency job ID is required}; shift 2 ;;
        --array) array=${2:?array specification is required}; shift 2 ;;
        --exclude) exclude=${2:?excluded node is required}; shift 2 ;;
        *) printf 'unknown FIR R2 submit argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$profile" in
    cpu) partition=CPU; cpus=2; memory=8G; walltime=00:30:00; gres="" ;;
    cpu-high) partition=cpu-high; cpus=8; memory=64G; walltime=1-00:00:00; gres="" ;;
    P100) partition=P100; cpus=8; memory=64G; walltime=02:00:00; gres=gpu:1 ;;
    L40S) partition=L40S; cpus=8; memory=64G; walltime=03:00:00; gres=gpu:1 ;;
    A100) partition=A100; cpus=8; memory=64G; walltime=12:00:00; gres=gpu:1 ;;
    H100) partition=H100; cpus=15; memory=128G; walltime=12:00:00; gres=gpu:1 ;;
    A100-H100) partition=A100,H100; cpus=12; memory=96G; walltime=12:00:00; gres=gpu:1 ;;
    *) printf 'unsupported FIR R2 profile: %s\n' "$profile" >&2; exit 2 ;;
esac

mkdir -p "$CODE_ROOT/slurm_logs/fir_r2_repair"
arguments=(
    --parsable
    --job-name="firr2_${stage}"
    --account=c2s
    --qos=normal
    --partition="$partition"
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$walltime"
    --output="$CODE_ROOT/slurm_logs/fir_r2_repair/%x_%A_%a.out"
    --error="$CODE_ROOT/slurm_logs/fir_r2_repair/%x_%A_%a.err"
    --export="ALL,DENOISENET_PROFILE=$profile"
)
[[ -z "$gres" ]] || arguments+=(--gres="$gres")
[[ -z "$dependency" ]] || arguments+=(--dependency="afterok:$dependency")
[[ -z "$array" ]] || arguments+=(--array="$array")
[[ -z "$exclude" ]] || arguments+=(--exclude="$exclude")
exec /usr/bin/sbatch "${arguments[@]}" \
    "$CODE_ROOT/scripts/slurm/jobs/fir_r2_repair.sbatch" \
    "$stage" configs/cgdr/subject_aware_diffusion_exploration_v2_fir_r2_repair.yaml
