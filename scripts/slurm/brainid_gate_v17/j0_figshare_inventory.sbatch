#!/usr/bin/env bash
#SBATCH --job-name=brainid_j0meta
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%j.out
#SBATCH --error=/home/infres/yinwang/denoiseNet_brainid_gate_v17/.slurm/brainid_gate_v17/%j.err
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_brainid_gate_v17
data=/projects/EEG-foundation-model/Longitudinal_ERP_Brainprint
mkdir -p "$repo/.slurm/brainid_gate_v17" "$repo/reports/slurm" "$data"
tmp="$data/.figshare_article.json.part"
curl -fL --retry 8 --retry-delay 10 --continue-at - \
  https://api.figshare.com/v2/articles/27201003 -o "$tmp"
mv "$tmp" "$data/figshare_article_27201003.json"
jq -r '["file_id","name","size_bytes","download_url","mtime"],(.files[]|[.id,.name,.size,.download_url,(.supplied_md5//"")])|@csv' \
  "$data/figshare_article_27201003.json" > "$data/source_inventory.csv"
jq '{id,title,doi,url_public_api,defined_type_name,published_date,modified_date,file_count:(.files|length),total_size_bytes:([.files[].size]|add)}' \
  "$data/figshare_article_27201003.json" > "$data/source_summary.json"
find "$data" -maxdepth 2 -type f -printf '%P,%s,%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$data/local_inventory_before_download.csv"
jq . "$data/source_summary.json"
