#!/usr/bin/env bash
#SBATCH --job-name=phy_hybrid
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked/results/cgdr/physiomotion_hybrid_masked/slurm/%A_%a.out
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p results/cgdr/physiomotion_hybrid_masked/slurm
STAGE="${STAGE:?}"; CONFIG=configs/cgdr/physiomotion_hybrid_masked.yaml
ARGS=(--config "$CONFIG" --stage "$STAGE" --run-dir "results/cgdr/physiomotion_hybrid_masked/runs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}")
if [[ -n "${FOLD:-}" ]]; then ARGS+=(--fold "$FOLD"); elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "$STAGE" != freeze && "$STAGE" != aggregate ]]; then ARGS+=(--fold "$SLURM_ARRAY_TASK_ID"); fi
if [[ -n "${SEED:-}" ]]; then ARGS+=(--seed "$SEED"); fi
if [[ "$STAGE" == technical || "$STAGE" == train || "$STAGE" == infer ]]; then
  source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate /home/infres/yinwang/anaconda3/envs/icml
else
  source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh; conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
fi
python -m eeg_cgdr.cli.physiomotion_hybrid_masked "${ARGS[@]}"
