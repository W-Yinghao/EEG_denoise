#!/usr/bin/env bash
set -euo pipefail
cd /home/infres/yinwang/denoiseNet_shu_task_phenotype
mkdir -p reports/slurm/shu_task_phenotype
job_map=reports/slurm/shu_task_phenotype_diffusion_job_ids.txt

tests=$(sbatch --parsable --job-name=shu_test --partition=CPU --cpus-per-task=2 --mem=8G --time=00:30:00 --output=reports/slurm/shu_task_phenotype/%A_test.out scripts/slurm/shu_task_phenotype/run_stage.sh tests)
freeze=$(sbatch --parsable --dependency=afterok:$tests --job-name=shu_freeze --partition=cpu-high --cpus-per-task=2 --mem=12G --time=01:00:00 --output=reports/slurm/shu_task_phenotype/%A_freeze.out scripts/slurm/shu_task_phenotype/run_stage.sh freeze)
materialize=$(sbatch --parsable --dependency=afterok:$freeze --job-name=shu_mat --partition=cpu-high --array=0-6%7 --cpus-per-task=2 --mem=10G --time=04:00:00 --output=reports/slurm/shu_task_phenotype/%A_%a_mat.out scripts/slurm/shu_task_phenotype/run_stage.sh materialize)
headroom=$(sbatch --parsable --dependency=afterok:$materialize --job-name=shu_head --partition=cpu-high --array=0-4%5 --cpus-per-task=8 --mem=48G --time=12:00:00 --output=reports/slurm/shu_task_phenotype/%A_%a_head.out scripts/slurm/shu_task_phenotype/run_stage.sh headroom)
aggregate=$(sbatch --parsable --dependency=afterok:$headroom --job-name=shu_agg --partition=cpu-high --cpus-per-task=2 --mem=16G --time=02:00:00 --output=reports/slurm/shu_task_phenotype/%A_agg.out scripts/slurm/shu_task_phenotype/run_stage.sh aggregate)
{
  printf '933143|J0_SOURCE_API_ATTEMPT|FAILED_BEFORE_NETWORK_DASH_PIPEFAIL_NO_DOWNLOAD\n'
  printf '933146|J0_DATALAKE_GLOBAL_SEARCH|CANCELLED_AFTER_ASSETS_LOCATED\n'
  printf '933153|J0_FOCUSED_SEARCH|FAILED_AWK_QUOTING_AFTER_FIRST_ASSET\n'
  printf '933164|J0_FOCUSED_SEARCH_RECOVERY|COMPLETED\n'
  printf '933172|J0_LMDB_CODEC_INSPECTION|COMPLETED\n'
  printf '933182|J0_DATALAKE_INVENTORY|COMPLETED\n'
  printf '933183|J0_SOURCE_METADATA|COMPLETED\n'
  printf '%s|J0_TARGETED_TESTS|SUBMITTED\n' "$tests"
  printf '%s|J0_PROTOCOL_MASK_FREEZE|AFTEROK_%s\n' "$freeze" "$tests"
  printf '%s|J0_DEVELOPMENT_MATERIALIZE_25|AFTEROK_%s\n' "$materialize" "$freeze"
  printf '%s|J1_HEADROOM_FOLDS_5|AFTEROK_%s\n' "$headroom" "$materialize"
  printf '%s|J1_AGGREGATE_ROUTE|AFTEROK_%s\n' "$aggregate" "$headroom"
} > "$job_map"
printf '%s %s %s %s %s\n' "$tests" "$freeze" "$materialize" "$headroom" "$aggregate"
