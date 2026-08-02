"""Static fail-closed Slurm routes for the matched conditional comparator."""

from __future__ import annotations

from pathlib import Path


CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")
CONFIG_PATH = Path("configs/cgdr/klados_stage3_conditional_diffusion_v3.yaml")


def test_conditional_training_is_three_scope_gpu_array_after_deterministic() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert '"stage3-conditional-diffusion"' in cli
    assert 'args.mode == "stage3-conditional-diffusion"' in cli
    assert "train_operator_conditioned_diffusion" in cli
    assert 'FROZEN_OPERATOR_SOURCES[' in cli
    assert '_array_task_index("train-conditional")' in cli
    assert "train-conditional)" in job
    assert '"${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$' in job
    assert (
        "train-conditional:L40S|train-conditional:A100|"
        "train-conditional:H100|train-conditional:V100-32GB|"
        "train-conditional:gpu-any)" in submitter
    )
    assert '"$array_spec" == \'0-2%3\'' in submitter
    assert '"$array_spec" =~ ^[0-2]$' in submitter
    assert (
        "conditional training requires --afterok with the deterministic v4 "
        "training array job ID" in submitter
    )
    assert "training_array_after_deterministic_command:" in config
    assert "--afterok DETERMINISTIC_V4_TRAIN_ARRAY_JOB_ID --array '0-2%3'" in config


def test_conditional_development_requires_optimizer_audit_dependency() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert "run_conditional_development_record" in cli
    assert '_array_task_index("conditional-development-record")' in cli
    assert '"${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$' in job
    assert '"$array_spec" == \'0-7%8\'' in submitter
    assert '"$dependency" =~ ^afterok:[0-9]+$' in submitter
    assert (
        "conditional development-record requires --afterok with the full "
        "conditional training array job ID" in submitter
    )
    assert "optimizer_step_audit_after_training_command:" in config
    assert "configs/cgdr/optimizer_step_audit_v3.yaml audit" in config
    assert "development_array_after_optimizer_audit_command:" in config
    assert "--afterok CONDITIONAL_V3_OPTIMIZER_AUDIT_JOB_ID --array '0-7%8'" in config


def test_conditional_aggregate_is_cpu_high_after_full_development_array() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert "aggregate_conditional_development" in cli
    assert "stage3-conditional-diffusion aggregate-development requires cpu-high" in job
    assert "aggregate-development:cpu-high)" in submitter
    assert "conditional aggregate-development does not accept an array" in submitter
    assert (
        "conditional aggregate-development requires --afterok for both "
        "conditional and deterministic full development array job IDs" in submitter
    )
    assert "aggregate_after_development_command:" in config
    assert "--afterok CONDITIONAL_V3_DEVELOPMENT_ARRAY_JOB_ID" in config
    assert "--afterok DETERMINISTIC_V4_DEVELOPMENT_ARRAY_JOB_ID" in config


def test_conditional_resume_and_scope_rules_fail_closed() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")

    branch = cli.split(
        'elif args.mode == "stage3-conditional-diffusion":', maxsplit=1
    )[1].split('elif args.mode == "eegdfus-benchmark":', maxsplit=1)[0]
    assert 'if training_result.status == "checkpointed_for_resume"' in branch
    assert "\n                75\n" in branch
    assert "historical-record" not in branch
    assert "aggregate-historical" not in branch
    assert "trap forward_usr1 USR1" in job
    assert 'kill -USR1 "$child_pid"' in job
    assert 'exit "$child_status"' in job
    assert 'light_sbatch_args+=(--signal="$checkpoint_signal")' in submitter
    assert "stage3-conditional-diffusion config is missing or unsafe" in submitter
