#!/usr/bin/env bash
#SBATCH --job-name=brainid17r_clean
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17r/.slurm/brainid_gate_v17r/%j.err
set -euo pipefail

repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17r
scratch_dir="$(mktemp -d /tmp/brainid_gate_v17r_clean.XXXXXX)"
cleanup() {
    case "$scratch_dir" in
        /tmp/brainid_gate_v17r_clean.*) rm -rf -- "$scratch_dir" ;;
        *) printf 'Refusing unsafe cleanup target: %s\n' "$scratch_dir" >&2 ;;
    esac
}
trap cleanup EXIT

source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
git -C "$repo" archive HEAD | tar -x -C "$scratch_dir"
cd "$scratch_dir"
export PYTHONPATH="$scratch_dir/src"
pytest -q tests/unit/test_brainid_gate_v17r.py
python -c 'import eeg_cgdr.experiments.brainid_gate_v17r; import eeg_cgdr.cli.brainid_gate_v17r'
