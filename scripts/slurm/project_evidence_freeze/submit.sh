#!/usr/bin/env bash
set -euo pipefail
repo=/home/infres/yinwang/denoiseNet_project_evidence_freeze
jobmap="$repo/reports/slurm/project_evidence_freeze_job_ids.txt"
mkdir -p "$repo/reports/slurm" "$repo/.slurm/project_evidence_freeze"
j0=$(sbatch --parsable "$repo/scripts/slurm/project_evidence_freeze/generate.sbatch")
j1=$(sbatch --parsable --dependency="afterok:$j0" "$repo/scripts/slurm/project_evidence_freeze/test.sbatch")
printf '%s|J0|cpu-high|generate registries, evidence tables, plots, and reports\n%s|J1|cpu|targeted tests and import|afterok:%s\n' "$j0" "$j1" "$j0" >> "$jobmap"
printf '%s %s\n' "$j0" "$j1"
