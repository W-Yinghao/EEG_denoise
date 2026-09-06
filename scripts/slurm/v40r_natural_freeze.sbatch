#!/usr/bin/env bash
#SBATCH --job-name=v40r-natural-freeze
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/home/infres/yinwang/slurm_logs/official_support_diffusion_v40r/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/official_support_diffusion_v40r/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_official_support_diffusion_v40r
cd "$root";export PYTHONPATH="$root/src:$root";export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
seeds=(20261010 20261011);fold=$((SLURM_ARRAY_TASK_ID/2));si=$((SLURM_ARRAY_TASK_ID%2));seed="${seeds[$si]}";source_root="${V40R_SOURCE_ROOT:?required}";run_id="natural_job_${SLURM_ARRAY_JOB_ID}"
exec /home/infres/yinwang/anaconda3/envs/icml/bin/python -m eeg_scad.cli.run_v40r natural --fold "$fold" --seed "$seed" --run-id "$run_id" --source-result "$source_root/fold_${fold}_seed_${seed}/result.json"
