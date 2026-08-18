#!/bin/bash
#SBATCH --job-name=s356-fetch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/yinwang/slurm_logs/iris/s356_fetch_%j.out
#SBATCH --error=/home/infres/yinwang/slurm_logs/iris/s356_fetch_%j.err
set -uo pipefail
cd /home/infres/yinwang/denoiseNet
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
# Loop until the quota window lets everything through (resumable, idempotent);
# 4-h backoff between attempts, bounded by the 24-h job limit — resubmit if needed.
for attempt in 1 2 3 4 5 6; do
  echo "=== fetch attempt $attempt $(date -u +%FT%TZ) ==="
  $PY scripts/iris/s356_acquire.py fetch
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('results/iris/s356/ext_fetch_report.json'))['complete'] else 1)"; then
    echo "COMPLETE"; exit 0
  fi
  sleep 14400
done
echo "INCOMPLETE after 6 attempts — resubmit"
