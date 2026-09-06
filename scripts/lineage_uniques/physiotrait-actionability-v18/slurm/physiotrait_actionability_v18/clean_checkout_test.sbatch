#!/usr/bin/env bash
#SBATCH --job-name=trait18_clean
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18/.slurm/physiotrait_actionability_v18/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_physiotrait_actionability_v18
scratch_dir="$(mktemp -d /tmp/physiotrait_v18_clean.XXXXXX)"
cleanup() {
    case "$scratch_dir" in
        /tmp/physiotrait_v18_clean.*) rm -rf -- "$scratch_dir" ;;
        *) printf 'Refusing unsafe cleanup target: %s\n' "$scratch_dir" >&2 ;;
    esac
}
trap cleanup EXIT
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
git -C "$repo" archive HEAD | tar -x -C "$scratch_dir"
cd "$scratch_dir"
export PYTHONPATH="$scratch_dir/src"
pytest -q tests/unit/test_physiotrait_actionability_v18.py
python -c 'import eeg_cgdr.experiments.physiotrait_actionability_v18; import eeg_cgdr.cli.physiotrait_actionability_v18'
