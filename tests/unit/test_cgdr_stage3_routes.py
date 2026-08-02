"""Static Slurm contracts for scope-isolated stage-3 training."""

from __future__ import annotations

from pathlib import Path


JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")
CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
CONFIG_PATH = Path("configs/cgdr/klados_stage3_deterministic_comparison.yaml")


def test_stage3_training_is_a_three_scope_gpu_array() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert "train-deterministic)" in job
    assert '"${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$' in job
    assert "training requires array task 0, 1 or 2" in job
    assert (
        "train-deterministic:L40S|train-deterministic:A100|"
        "train-deterministic:H100|train-deterministic:V100-32GB|"
        "train-deterministic:gpu-any)" in submitter
    )
    assert '"$array_spec" == \'0-2%3\'' in submitter
    assert '"$array_spec" =~ ^[0-2]$' in submitter
    assert "FROZEN_OPERATOR_SOURCES[" in cli
    assert '_array_task_index("train-deterministic")' in cli
    assert "operator_source=operator_scope" in cli


def test_stage3_development_waits_for_the_full_training_array() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert '"${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$' in job
    assert '"$array_spec" == \'0-7%8\'' in submitter
    assert '"$dependency" =~ ^afterok:[0-9]+$' in submitter
    assert "requires --afterok with the full training array job ID" in submitter
    assert "training_array_command:" in config
    assert "--array '0-2%3'" in config
    assert "development_array_after_training_command:" in config
    assert "--afterok TRAIN_ARRAY_JOB_ID --array '0-7%8'" in config


def test_stage3_array_route_preserves_run_identity_and_usr1_exit_status() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert "train-deterministic:gpu-any)" in submitter
    assert 'light_sbatch_args+=(--signal="$checkpoint_signal")' in submitter
    assert 'task_suffix="_${SLURM_ARRAY_TASK_ID}"' in job
    assert '"$SLURM_ARRAY_TASK_ID" > "$run_dir/slurm_array_task_id.txt"' in job
    assert "trap forward_usr1 USR1" in job
    assert "kill -USR1 \"$child_pid\"" in job
    assert 'exit "$child_status"' in job
    training_branch = cli.split("training_result =", maxsplit=1)[1].split(
        'elif args.stage in {"development-record", "historical-record"}',
        maxsplit=1,
    )[0]
    assert 'if training_result.status == "checkpointed_for_resume"' in training_branch
    assert "\n                    75\n" in training_branch
