#!/bin/bash
#SBATCH --job-name=m13-prior
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/flagship_m0/%A_%a.err
set -euo pipefail
root=/home/infres/yinwang/denoiseNet_flagship_m0
export PYTHONPATH="$root/src:$root"
cd "$root"
run=${M13_RUN:?M13_RUN required}; seed=${M13_SEED:?M13_SEED required}
py=/home/infres/yinwang/anaconda3/envs/icml/bin/python
"$py" -m eeg_chart.run_m13 w2-train --run "$run" --seed "$seed" --updates "${M13_UPDATES:-60000}"
"$py" -m eeg_chart.run_m13 w2-eval --run "$run" --seed "$seed"
