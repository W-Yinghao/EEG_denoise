#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
stage=$1;ROOT=/home/infres/yinwang/denoiseNet_bci2b_eog_residual_v11_1;CPU=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python;GPU=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "$ROOT";export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}";jid=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID};run="$ROOT/results/cgdr/bci2b_eog_residual_v11_1/runs/$stage/job_${jid}${SLURM_ARRAY_TASK_ID:+/task_${SLURM_ARRAY_TASK_ID}}";mkdir -p "$run"
case "$stage" in
 tests) "$CPU" -m pytest -q tests/unit/test_bci2b_eog_residual_v11_1.py | tee "$run/pytest.txt" ;;
 clean-import) d=${SLURM_TMPDIR:-$run/clean};mkdir -p "$d";git archive HEAD|tar -x -C "$d";PYTHONPATH="$d/src:$d" "$CPU" -c 'import eeg_cgdr.experiments.bci2b_eog_residual_v11_1; print("V11.1 clean import passed")' ;;
 materialize|forensic|freeze-contract|evaluate|aggregate|finalize) exec "$CPU" -m eeg_cgdr.cli.bci2b_eog_residual_v11_1 --config configs/cgdr/bci2b_eog_residual_v11_1.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
 *) exec "$GPU" -m eeg_cgdr.cli.bci2b_eog_residual_v11_1 --config configs/cgdr/bci2b_eog_residual_v11_1.yaml --stage "$stage" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --run-dir "$run" ;;
esac
