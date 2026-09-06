#!/usr/bin/env bash
#SBATCH --job-name=v37t-k16
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-14%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/taas_waveform_v37t/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/taas_waveform_v37t/%A_%a.err

set -euo pipefail
readonly code_root=/home/infres/yinwang/denoiseNet_taas_waveform_v37t
readonly python_bin=/home/infres/yinwang/anaconda3/envs/icml/bin/python
mkdir -p /home/infres/yinwang/slurm_logs/taas_waveform_v37t
cd "$code_root"
export DENOISENET_CODE_ROOT="$code_root"
export PYTHONPATH="$code_root/src:$code_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
exec "$python_bin" -m eeg_scad.cli.run_v37t stochastic --index "$SLURM_ARRAY_TASK_ID"

