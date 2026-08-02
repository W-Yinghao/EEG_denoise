"""Static Slurm route contract for the six-checkpoint CPU audit."""

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/cgdr/optimizer_step_audit.yaml")
CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")


def test_optimizer_step_audit_has_one_cpu_non_array_route() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")
    submitter = SUBMIT_PATH.read_text(encoding="utf-8")

    assert '"optimizer-step-audit"' in cli
    assert 'args.mode == "optimizer-step-audit"' in cli
    assert "run_optimizer_step_audit" in cli
    assert "optimizer-step-audit)" in job
    assert "optimizer-step-audit audit requires cpu" in job
    assert "optimizer-step-audit rejects array tasks" in job
    assert "tests/unit/test_cgdr_optimizer_step_audit.py" in job
    assert "tests/unit/test_cgdr_optimizer_step_audit_routes.py" in job
    assert "optimizer-step-audit audit requires cpu" in submitter
    assert "optimizer-step-audit rejects arrays" in submitter


def test_optimizer_step_audit_config_names_exact_six_endpoints() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    checkpoints = config["checkpoints"]

    assert config["expected_optimizer_step"] == 6000
    assert config["output_root"] == "results/cgdr/optimizer_step_audit"
    assert set(checkpoints) == {"deterministic_best", "conditional_final"}
    assert set(checkpoints["deterministic_best"]) == {
        "population_projector",
        "matching_p0",
        "query_derived_oracle_projector",
    }
    assert all(
        Path(path).name == "best.pt"
        for path in checkpoints["deterministic_best"].values()
    )
    assert all(
        Path(path).name == "final.pt"
        for path in checkpoints["conditional_final"].values()
    )
    assert config["execution"]["command"] == (
        "scripts/slurm/submit.sh cpu cgdr optimizer-step-audit "
        "configs/cgdr/optimizer_step_audit.yaml audit"
    )


def test_replacement_v3_audit_targets_new_conditional_root() -> None:
    path = Path("configs/cgdr/optimizer_step_audit_v3.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    conditional = config["checkpoints"]["conditional_final"]
    assert config["audit_id"] == "frozen_stage3_optimizer_steps_v3"
    assert config["output_root"] == "results/cgdr/optimizer_step_audit_v3"
    assert all("matched_v3/checkpoints" in value for value in conditional.values())
    assert all(Path(value).name == "final.pt" for value in conditional.values())
