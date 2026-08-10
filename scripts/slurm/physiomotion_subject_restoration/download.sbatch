#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration
DATA=/projects/EEG-foundation-model/PhysioMotion_Artifact
OUT="$ROOT/results/cgdr/physiomotion_subject_restoration/inventory"
mkdir -p "$DATA" "$OUT"
available_kb=$(df --output=avail /projects/EEG-foundation-model | tail -1)
(( available_kb > 60 * 1024 * 1024 )) || { echo "less than 60 GiB free" >&2; exit 3; }
/home/infres/yinwang/.local/bin/aws s3 sync s3://openneuro.org/ds006386 "$DATA" --no-sign-request --only-show-errors
find "$DATA" -maxdepth 4 -type f -printf '%s|%p\n' | sort > "$OUT/file_inventory_after_download.txt"
edf=$(find "$DATA" -type f -name '*_eeg.edf' | wc -l)
events=$(find "$DATA" -type f -name '*_events.tsv' | wc -l)
annotations=$(find "$DATA/derivatives/Manual_Annotations" -type f -name '*.csv' 2>/dev/null | wc -l)
printf 'edf=%s\nevents=%s\nmanual_annotations=%s\n' "$edf" "$events" "$annotations" > "$OUT/download_coverage.txt"
[[ "$edf" -eq 180 && "$events" -eq 180 && "$annotations" -eq 180 ]]
