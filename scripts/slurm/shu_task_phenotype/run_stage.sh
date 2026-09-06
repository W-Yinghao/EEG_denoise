#!/usr/bin/env bash
set -euo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh
conda activate /home/infres/yinwang/anaconda3/envs/eeg2025
cd /home/infres/yinwang/denoiseNet_shu_task_phenotype
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

stage="${1:?stage required}"
case "$stage" in
  tests)
    python -m pytest -q tests/unit/test_shu_task_phenotype.py
    ;;
  freeze)
    python -m eeg_cgdr.experiments.shu_task_phenotype freeze
    ;;
  materialize)
    subject=$((SLURM_ARRAY_TASK_ID + 1))
    while [[ "$subject" -le 25 ]]; do
      python -m eeg_cgdr.experiments.shu_task_phenotype materialize --subject "$subject"
      subject=$((subject + 7))
    done
    ;;
  headroom)
    python -m eeg_cgdr.experiments.shu_task_phenotype headroom-fold --fold "$SLURM_ARRAY_TASK_ID"
    ;;
  aggregate)
    python -m eeg_cgdr.experiments.shu_task_phenotype aggregate
    ;;
  report)
    python -m eeg_cgdr.experiments.shu_task_phenotype report
    ;;
  *)
    echo "unknown stage: $stage" >&2; exit 2
    ;;
esac
