#!/usr/bin/env bash
#SBATCH --job-name=brainid_download
#SBATCH --partition=CPU
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%A_%a.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%A_%a.err
set -euo pipefail
data=/projects/EEG-foundation-model/Longitudinal_ERP_Brainprint
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17
mkdir -p "$data/files" "$repo/.slurm/brainid_gate_v17"
count=$(jq '.files|length' "$data/figshare_article_27201003.json")
for ((index=SLURM_ARRAY_TASK_ID; index<count; index+=4)); do
  record=$(jq -c ".files[$index]" "$data/figshare_article_27201003.json")
  name=$(jq -r .name <<<"$record")
  url=$(jq -r .download_url <<<"$record")
  expected=$(jq -r .size <<<"$record")
  part="$data/files/$name.part"
  final="$data/files/$name"
  if [[ -f "$final" ]] && [[ $(stat -c %s "$final") -eq "$expected" ]]; then
    printf 'already complete: %s %s\n' "$name" "$expected"
    continue
  fi
  curl -fL --retry 12 --retry-delay 15 --continue-at - "$url" -o "$part"
  actual=$(stat -c %s "$part")
  if [[ "$actual" -ne "$expected" ]]; then
    printf 'size mismatch: %s expected=%s actual=%s\n' "$name" "$expected" "$actual" >&2
    exit 2
  fi
  mv "$part" "$final"
  touch -r "$data/figshare_article_27201003.json" "$final"
  printf 'complete: %s %s\n' "$name" "$actual"
done
