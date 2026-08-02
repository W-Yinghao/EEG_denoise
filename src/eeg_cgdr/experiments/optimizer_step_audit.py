"""Retrospective AdamW step audit for the frozen Stage-3 endpoints.

This module intentionally loads only the six checkpoint paths named by the
targeted audit configuration.  It does not instantiate a model, read EEG, or
resume training.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eeg_cgdr.training import load_training_checkpoint


OPERATOR_SCOPES = (
    "population_projector",
    "matching_p0",
    "query_derived_oracle_projector",
)
_CHECKPOINT_GROUPS = {
    "deterministic_best": "best.pt",
    "conditional_final": "final.pt",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _step_value(value: Any) -> int:
    """Return one exact non-negative integer optimizer step."""

    numel = getattr(value, "numel", None)
    if callable(numel):
        if int(numel()) != 1:
            raise ValueError("optimizer step tensor must contain one value")
        value = value.item()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("optimizer step cannot be boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("optimizer step is not numeric") from error
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError("optimizer step must be a finite non-negative integer")
    return int(numeric)


def _small_json_value(value: Any) -> Any:
    """Serialize the small GradScaler state without copying tensor payloads."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _small_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"type": type(value).__name__, "length": len(value)}
        return [_small_json_value(item) for item in value]
    numel = getattr(value, "numel", None)
    if callable(numel):
        element_count = int(numel())
        if element_count == 1:
            return _small_json_value(value.item())
        shape = getattr(value, "shape", ())
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in shape],
            "element_count": element_count,
        }
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _small_json_value(item())
        except (TypeError, ValueError):
            pass
    return {"type": type(value).__name__}


def audit_checkpoint_payload(
    payload: Mapping[str, Any], *, expected_step: int
) -> dict[str, Any]:
    """Audit one already-loaded checkpoint using only its training state."""

    failures: list[str] = []
    try:
        checkpoint_step = _step_value(payload.get("step"))
    except ValueError as error:
        checkpoint_step = None
        failures.append(f"invalid checkpoint step: {error}")
    if checkpoint_step is not None and checkpoint_step != expected_step:
        failures.append(
            f"checkpoint step is {checkpoint_step}, expected {expected_step}"
        )

    optimizer_state = payload.get("optimizer_state")
    state: Mapping[Any, Any] = {}
    parameter_ids: list[Any] = []
    if not isinstance(optimizer_state, Mapping):
        failures.append("optimizer_state is missing or is not a mapping")
    else:
        raw_state = optimizer_state.get("state")
        if isinstance(raw_state, Mapping):
            state = raw_state
        else:
            failures.append("optimizer_state.state is missing or is not a mapping")
        param_groups = optimizer_state.get("param_groups")
        if not isinstance(param_groups, Sequence) or isinstance(
            param_groups, (str, bytes)
        ):
            failures.append("optimizer_state.param_groups is missing or invalid")
        else:
            for group_index, group in enumerate(param_groups):
                if not isinstance(group, Mapping):
                    failures.append(f"optimizer parameter group {group_index} is invalid")
                    continue
                params = group.get("params")
                if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
                    failures.append(
                        f"optimizer parameter group {group_index} has invalid params"
                    )
                    continue
                parameter_ids.extend(params)

    if not state:
        failures.append("optimizer per-parameter state is empty")
    if not parameter_ids:
        failures.append("optimizer parameter groups contain no parameters")
    if len(set(parameter_ids)) != len(parameter_ids):
        failures.append("optimizer parameter groups contain duplicate parameters")

    state_ids = set(state)
    parameter_id_set = set(parameter_ids)
    missing_state_ids = parameter_id_set - state_ids
    unexpected_state_ids = state_ids - parameter_id_set
    if missing_state_ids:
        failures.append(
            f"{len(missing_state_ids)} optimizer parameters lack AdamW state"
        )
    if unexpected_state_ids:
        failures.append(
            f"{len(unexpected_state_ids)} AdamW states are absent from parameter groups"
        )

    steps: list[int] = []
    missing_step_count = 0
    invalid_step_count = 0
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            missing_step_count += 1
            continue
        try:
            steps.append(_step_value(parameter_state["step"]))
        except ValueError:
            invalid_step_count += 1
    if missing_step_count:
        failures.append(f"{missing_step_count} AdamW parameter states lack step")
    if invalid_step_count:
        failures.append(f"{invalid_step_count} AdamW step counters are invalid")
    unique_steps = sorted(set(steps))
    if steps and unique_steps != [expected_step]:
        failures.append(
            "AdamW parameter steps are not all equal to "
            f"{expected_step}: observed {unique_steps}"
        )

    scaler_state = payload.get("scaler_state")
    scaler_mapping = scaler_state if isinstance(scaler_state, Mapping) else None
    return {
        "status": "passed" if not failures else "failed",
        "checkpoint_step": checkpoint_step,
        "optimizer_parameter_count": len(parameter_id_set),
        "optimizer_state_count": len(state),
        "adam_step_count": len(steps),
        "adam_step_min": min(steps) if steps else None,
        "adam_step_max": max(steps) if steps else None,
        "adam_step_unique": unique_steps,
        "missing_parameter_state_count": len(missing_state_ids),
        "unexpected_parameter_state_count": len(unexpected_state_ids),
        "missing_step_count": missing_step_count,
        "invalid_step_count": invalid_step_count,
        "scaler_state": {
            "present": scaler_state is not None,
            "mapping": scaler_mapping is not None,
            "empty": len(scaler_mapping) == 0 if scaler_mapping is not None else None,
            "keys": sorted(str(key) for key in scaler_mapping)
            if scaler_mapping is not None
            else [],
            "values": _small_json_value(scaler_state),
        },
        "failure_reasons": failures,
    }


def _checkpoint_specs(config: Mapping[str, Any]) -> list[dict[str, str]]:
    groups = _mapping(config.get("checkpoints"), "checkpoints")
    if set(groups) != set(_CHECKPOINT_GROUPS):
        raise ValueError(
            "optimizer-step audit requires exactly deterministic_best and "
            "conditional_final checkpoint groups"
        )
    specs: list[dict[str, str]] = []
    for family, required_name in _CHECKPOINT_GROUPS.items():
        paths = _mapping(groups[family], family)
        if set(paths) != set(OPERATOR_SCOPES):
            raise ValueError(
                f"{family} must name exactly the three frozen operator scopes"
            )
        for operator_scope in OPERATOR_SCOPES:
            path = Path(str(paths[operator_scope]))
            if path.name != required_name:
                raise ValueError(
                    f"{family}/{operator_scope} must point to {required_name}"
                )
            specs.append(
                {
                    "family": family,
                    "operator_scope": operator_scope,
                    "path": str(path),
                }
            )
    resolved = [str(Path(spec["path"]).resolve()) for spec in specs]
    if len(specs) != 6 or len(set(resolved)) != 6:
        raise ValueError("optimizer-step audit requires six distinct checkpoints")
    return specs


def _markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen Stage-3 optimizer-step audit",
        "",
        "This retrospective audit only reads three deterministic `best.pt` and "
        "three conditional `final.pt` checkpoints. It does not read EEG or resume "
        "training.",
        "",
        f"Status: `{result['status']}`. Expected AdamW step: "
        f"`{result['expected_optimizer_step']}`.",
        "",
        "| Family | Operator scope | Global step | Adam states | Min | Max | "
        "Unique | Scaler | Status |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    failure_lines: list[str] = []
    for checkpoint in result["checkpoints"]:
        scaler = checkpoint.get("scaler_state", {})
        scaler_label = (
            "missing"
            if not scaler.get("present")
            else "empty"
            if scaler.get("empty")
            else ", ".join(scaler.get("keys", []))
        )
        lines.append(
            "| {family} | {operator_scope} | {checkpoint_step} | "
            "{adam_step_count} | {adam_step_min} | {adam_step_max} | "
            "{adam_step_unique} | {scaler} | {status} |".format(
                scaler=scaler_label,
                **checkpoint,
            )
        )
        for reason in checkpoint.get("failure_reasons", []):
            failure_lines.append(
                f"- `{checkpoint['family']}/{checkpoint['operator_scope']}`: {reason}"
            )
    if failure_lines:
        lines.extend(["", "## Failures", "", *failure_lines])
    lines.extend(
        [
            "",
            f"Checked `{result['checkpoint_count_checked']}` of exactly "
            f"`{result['checkpoint_count_expected']}` configured checkpoints.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(directory: Path, result: Mapping[str, Any], report: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "optimizer_step_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "optimizer_step_audit.md").write_text(report, encoding="utf-8")


def run_optimizer_step_audit(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Load the frozen six endpoints and fail closed on any step mismatch."""

    expected_step = int(config.get("expected_optimizer_step", 0))
    if expected_step <= 0:
        raise ValueError("expected_optimizer_step must be positive")
    specs = _checkpoint_specs(config)
    checkpoints: list[dict[str, Any]] = []
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_file():
            audited = {
                "status": "failed",
                "checkpoint_step": None,
                "optimizer_parameter_count": 0,
                "optimizer_state_count": 0,
                "adam_step_count": 0,
                "adam_step_min": None,
                "adam_step_max": None,
                "adam_step_unique": [],
                "missing_parameter_state_count": 0,
                "unexpected_parameter_state_count": 0,
                "missing_step_count": 0,
                "invalid_step_count": 0,
                "scaler_state": {
                    "present": False,
                    "mapping": False,
                    "empty": None,
                    "keys": [],
                    "values": None,
                },
                "failure_reasons": ["checkpoint file is missing"],
            }
        else:
            try:
                payload = load_training_checkpoint(path, map_location="cpu")
                audited = audit_checkpoint_payload(
                    payload, expected_step=expected_step
                )
            except Exception as error:
                # Preserve a machine-readable failure even for a truncated or
                # otherwise unreadable trusted local checkpoint. SystemExit,
                # KeyboardInterrupt, and scheduler signals still propagate.
                audited = {
                    "status": "failed",
                    "checkpoint_step": None,
                    "optimizer_parameter_count": 0,
                    "optimizer_state_count": 0,
                    "adam_step_count": 0,
                    "adam_step_min": None,
                    "adam_step_max": None,
                    "adam_step_unique": [],
                    "missing_parameter_state_count": 0,
                    "unexpected_parameter_state_count": 0,
                    "missing_step_count": 0,
                    "invalid_step_count": 0,
                    "scaler_state": {
                        "present": False,
                        "mapping": False,
                        "empty": None,
                        "keys": [],
                        "values": None,
                    },
                    "failure_reasons": [
                        f"checkpoint could not be loaded: {type(error).__name__}: {error}"
                    ],
                }
        checkpoints.append(
            {
                **spec,
                "path": str(path.resolve()),
                **audited,
            }
        )

    passed = len(checkpoints) == 6 and all(
        item["status"] == "passed" for item in checkpoints
    )
    output_root = Path(
        str(config.get("output_root", "results/cgdr/optimizer_step_audit"))
    )
    result: dict[str, Any] = {
        "audit_id": str(config.get("audit_id", "frozen_stage3_optimizer_steps_v1")),
        "status": (
            "passed_exact_six_checkpoints_at_expected_optimizer_step"
            if passed
            else "failed_optimizer_step_contract"
        ),
        "expected_optimizer_step": expected_step,
        "checkpoint_count_expected": 6,
        "checkpoint_count_checked": len(checkpoints),
        "checkpoint_count_passed": sum(
            item["status"] == "passed" for item in checkpoints
        ),
        "no_eeg_loaded": True,
        "training_resume_status": (
            "not_audited_from_checkpoint; validated separately from each "
            "training result summary"
        ),
        "checkpoints": checkpoints,
        "output_root": str(output_root.resolve()),
        "run_dir": str(run_dir.resolve()),
    }
    report = _markdown_report(result)
    _write_outputs(output_root, result, report)
    _write_outputs(run_dir, result, report)
    return result


__all__ = [
    "OPERATOR_SCOPES",
    "audit_checkpoint_payload",
    "run_optimizer_step_audit",
]
