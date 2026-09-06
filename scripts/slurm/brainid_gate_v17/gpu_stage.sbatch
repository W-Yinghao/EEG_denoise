#!/usr/bin/env bash
#SBATCH --job-name=brainid_gpu
#SBATCH --partition=A100,H100,L40S
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%A_%a.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%A_%a.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/icml
cd "$repo"; export PYTHONPATH="$repo/src"; export CUBLAS_WORKSPACE_CONFIG=:4096:8
stage="${STAGE:?}"; task="$SLURM_ARRAY_TASK_ID"; run="$repo/results/cgdr/brainid_gate_v17/server_runs/$stage/task_${task}"
python -m eeg_cgdr.cli.brainid_gate_v17 --config configs/cgdr/brainid_gate_v17.yaml --stage "$stage" --index "$task" --run-dir "$run"
