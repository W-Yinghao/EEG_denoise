"""Focused pure-state tests for the retrospective optimizer audit."""

from __future__ import annotations

import json
from pathlib import Path

from eeg_cgdr.experiments.optimizer_step_audit import (
    OPERATOR_SCOPES,
    audit_checkpoint_payload,
    run_optimizer_step_audit,
)


def _payload(steps: tuple[int, ...] = (6000, 6000)) -> dict[str, object]:
    parameter_ids = list(range(len(steps)))
    return {
        "step": 6000,
        "optimizer_state": {
            "state": {
                parameter_id: {
                    "step": step,
                    "exp_avg": "synthetic-not-inspected",
                    "exp_avg_sq": "synthetic-not-inspected",
                }
                for parameter_id, step in zip(parameter_ids, steps)
            },
            "param_groups": [{"params": parameter_ids}],
        },
        "scaler_state": {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 17,
        },
    }


def test_exact_per_parameter_steps_and_scaler_are_reported() -> None:
    result = audit_checkpoint_payload(_payload(), expected_step=6000)

    assert result["status"] == "passed"
    assert result["optimizer_parameter_count"] == 2
    assert result["adam_step_count"] == 2
    assert result["adam_step_min"] == 6000
    assert result["adam_step_max"] == 6000
    assert result["adam_step_unique"] == [6000]
    assert result["scaler_state"]["values"]["scale"] == 65536.0


def test_mixed_or_missing_parameter_adam_steps_fail_closed() -> None:
    mixed = audit_checkpoint_payload(_payload((5999, 6000)), expected_step=6000)
    assert mixed["status"] == "failed"
    assert mixed["adam_step_min"] == 5999
    assert mixed["adam_step_max"] == 6000
    assert mixed["adam_step_unique"] == [5999, 6000]

    missing = _payload()
    del missing["optimizer_state"]["state"][1]
    missing_result = audit_checkpoint_payload(missing, expected_step=6000)
    assert missing_result["status"] == "failed"
    assert missing_result["missing_parameter_state_count"] == 1

    empty_result = audit_checkpoint_payload(_payload(()), expected_step=6000)
    assert empty_result["status"] == "failed"
    assert empty_result["adam_step_count"] == 0
    assert "optimizer per-parameter state is empty" in empty_result["failure_reasons"]


def test_runner_loads_exactly_three_best_and_three_final_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    paths: dict[str, dict[str, str]] = {
        "deterministic_best": {},
        "conditional_final": {},
    }
    for family, filename in (
        ("deterministic_best", "best.pt"),
        ("conditional_final", "final.pt"),
    ):
        for scope in OPERATOR_SCOPES:
            path = tmp_path / family / scope / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic loader sentinel\n", encoding="utf-8")
            paths[family][scope] = str(path)

    loaded: list[Path] = []

    def fake_loader(path: Path, *, map_location: str):
        assert map_location == "cpu"
        loaded.append(Path(path))
        return _payload()

    monkeypatch.setattr(
        "eeg_cgdr.experiments.optimizer_step_audit.load_training_checkpoint",
        fake_loader,
    )
    output_root = tmp_path / "result"
    run_dir = tmp_path / "run"
    result = run_optimizer_step_audit(
        {
            "audit_id": "synthetic_test",
            "expected_optimizer_step": 6000,
            "output_root": str(output_root),
            "checkpoints": paths,
        },
        run_dir=run_dir,
    )

    assert result["status"] == "passed_exact_six_checkpoints_at_expected_optimizer_step"
    assert len(loaded) == 6
    assert {path.name for path in loaded} == {"best.pt", "final.pt"}
    assert sum(path.name == "best.pt" for path in loaded) == 3
    assert sum(path.name == "final.pt" for path in loaded) == 3
    for directory in (output_root, run_dir):
        payload = json.loads(
            (directory / "optimizer_step_audit.json").read_text(encoding="utf-8")
        )
        assert payload["checkpoint_count_checked"] == 6
        assert (directory / "optimizer_step_audit.md").is_file()
