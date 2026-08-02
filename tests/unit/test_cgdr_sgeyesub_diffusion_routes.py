"""Static Slurm/CLI contracts for the frozen natural-SGE comparison."""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/cgdr/sgeyesub_diffusion_incremental.yaml")
CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")


def test_sgeyesub_diffusion_cli_routes_all_frozen_stages() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert '"sgeyesub-diffusion",' in cli
    assert "run_sgeyesub_diffusion_cpu_validation" in cli
    assert "run_sgeyesub_diffusion_integration" in cli
    assert "run_sgeyesub_diffusion_fold" in cli
    assert "aggregate_sgeyesub_diffusion_partition" in cli
    assert '_array_task_index(args.stage)' in cli
    assert (
        'return_code = 75 if result["status"] == "checkpointed_for_resume" else 0'
        in cli
    )
    assert '"diffusion-incremental-decision-v2",' in cli
    assert "run_diffusion_incremental_decision_v2" in cli


def test_sgeyesub_diffusion_job_uses_registered_environments_and_arrays() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")

    assert "sgeyesub-diffusion)" in job
    assert "sgeyesub-diffusion cpu-tests requires cpu" in job
    assert "sgeyesub development-fold requires array task 0 through 9" in job
    assert "sgeyesub evaluation-fold requires array task 0 through 14" in job
    assert 'python_bin="$EEG_ENV/bin/python"' in job
    assert 'python_bin="$GPU_ENV/bin/python"' in job
    assert "tests/unit/test_cgdr_sgeyesub_diffusion.py" in job
    assert "tests/unit/test_cgdr_sgeyesub_diffusion_runner.py" in job
    assert "tests/unit/test_cgdr_diffusion_incremental_decision_v2.py" in job
    assert '"$GPU_ENV/bin/python" -m pytest' in job
    assert "trap forward_usr1 USR1" in job


def test_submitter_freezes_array_shape_profiles_and_dependencies() -> None:
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")

    assert "sgeyesub-diffusion requires CONFIG STAGE" in submitter
    assert "development-fold:V100-32GB" in submitter
    assert "evaluation-fold:V100-32GB" in submitter
    assert "--array 0-9%%8 or one retry index 0-9" in submitter
    assert "--array 0-14%%8 or one retry index 0-14" in submitter
    assert "full sgeyesub development-fold array requires an afterok" in submitter
    assert "full sgeyesub evaluation-fold array requires the development" in submitter
    assert "diffusion-incremental-decision-v2 requires afterok dependencies" in submitter


def test_config_keeps_full_scientific_denominators_and_concurrency() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert len(config["split"]["development_folds"]) == 10
    assert len(config["split"]["evaluation_folds"]) == 15
    assert config["evaluation"]["compatible_performance_denominator"] == 43
    assert config["evaluation"]["availability_denominator"] == 44
    assert config["execution_plan"]["development_fold_array"] == (
        "10_tasks_maximum_8_concurrent"
    )
    assert config["execution_plan"]["evaluation_fold_array"] == (
        "15_tasks_maximum_8_concurrent"
    )
