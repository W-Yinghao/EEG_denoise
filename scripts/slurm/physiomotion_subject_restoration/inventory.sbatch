#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration
DATA=/projects/EEG-foundation-model/PhysioMotion_Artifact
OUT="$ROOT/results/cgdr/physiomotion_subject_restoration/inventory"
mkdir -p "$OUT"
{
  date -u +%FT%TZ
  df -h /projects/EEG-foundation-model
  find /projects/EEG-foundation-model -maxdepth 3 -type d \( -iname '*physiomotion*' -o -iname 'ds006386*' \) -print
  for tool in aws datalad git-annex deno openneuro; do command -v "$tool" || true; done
} > "$OUT/environment.txt"
if [[ -d "$DATA" ]]; then
  find "$DATA" -maxdepth 4 -type f -printf '%s|%p\n' | sort > "$OUT/file_inventory.txt"
  printf 'present\n' > "$OUT/status.txt"
else
  printf 'missing\n' > "$OUT/status.txt"
  if command -v aws >/dev/null 2>&1; then
    aws s3 ls s3://openneuro.org/ds006386/ --recursive --no-sign-request > "$OUT/openneuro_s3_listing.txt"
  fi
fi
