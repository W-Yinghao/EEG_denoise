"""Static contracts for the subject-calibrated artifact experiment route."""

from pathlib import Path


CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")


def test_subject_artifact_cli_routes_every_frozen_stage() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert '"subject-artifact",' in cli
    assert 'args.mode == "subject-artifact"' in cli
    for stage in (
        "j0-audit",
        "j1-cpu",
        "validity",
        "train",
        "evaluate",
        "aggregate",
        "finalize",
    ):
        assert f'"{stage}"' in cli
    assert "subject_calibrated_artifact_runner import" in cli
    assert "run_stage(config, args.run_dir, args.stage, task_index)" in cli
    assert "_optional_array_task_index()" in cli


def test_subject_artifact_job_uses_fixed_profiles_and_array_scope() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    route = job.split("    subject-artifact)", maxsplit=1)[1].split(
        "    *) printf 'unknown CGDR mode", maxsplit=1
    )[0]

    assert "j0-audit:cpu|j1-cpu:cpu|finalize:cpu" in route
    assert "aggregate:cpu-high" in route
    assert "validity:L40S|validity:A100|validity:H100" in route
    assert "train:L40S|train:A100|train:H100" in route
    assert "evaluate:L40S|evaluate:A100|evaluate:H100" in route
    assert "V100" not in route
    assert "gpu-any" not in route
    assert "subject-artifact %s rejects array tasks" in route
    assert 'train|evaluate)' in route
    assert 'python_bin="$EEG_ENV/bin/python"' in route
    assert 'python_bin="$GPU_ENV/bin/python"' in route


def test_subject_artifact_j0_combines_unit_and_shell_validation_once() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    j0_block = job.split(
        'if [[ "$mode" == subject-artifact && "$stage" == j0-audit ]]',
        maxsplit=1,
    )[1].split("\nfi", maxsplit=1)[0]

    assert "/usr/bin/bash -n scripts/slurm/submit.sh" in j0_block
    assert "scripts/slurm/jobs/cgdr.sbatch" in j0_block
    assert 'pytest -q tests/unit/test_cgdr_*.py' in j0_block
    assert j0_block.count('pytest -q tests/unit/test_cgdr_*.py') == 1


def test_subject_artifact_submitter_is_fail_closed() -> None:
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    route = submitter.split(
        'elif [[ "${payload_args[0]}" == subject-artifact ]]', maxsplit=1
    )[1].split("\n        else\n", maxsplit=1)[0]

    assert "subject-artifact requires CONFIG STAGE" in route
    assert "j0-audit:cpu|j1-cpu:cpu|finalize:cpu|aggregate:cpu-high" in route
    assert "validity:L40S|validity:A100|validity:H100" in route
    assert "train:L40S|train:A100|train:H100" in route
    assert "evaluate:L40S|evaluate:A100|evaluate:H100" in route
    assert "V100" not in route
    assert "gpu-any" not in route
    assert "subject-artifact %s rejects arrays" in route
    assert "subject-artifact config must be inside the code root" in route
    assert "subject-artifact config is missing or unsafe" in route
