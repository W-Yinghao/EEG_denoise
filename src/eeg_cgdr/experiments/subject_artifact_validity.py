"""Pure-numeric V0--V3 checks for subject-calibrated artifact models.

The functions in this module do not read files, choose thresholds, or mutate
experiment state.  Every scientific limit comes from the supplied config and
every result is JSON-serializable for the stage runner to persist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np


Result = dict[str, Any]
Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], Result]


class _InvalidValidityInput(ValueError):
    """Internal signal converted to a machine-readable blocked result."""


def _level_config(config: Mapping[str, Any], level: str) -> Mapping[str, Any]:
    validity = config.get("validity", config)
    if not isinstance(validity, Mapping) or level not in validity:
        raise _InvalidValidityInput(f"missing configured validity level {level}")
    level_config = validity[level]
    if not isinstance(level_config, Mapping):
        raise _InvalidValidityInput(f"configured validity level {level} is not a mapping")
    return level_config


def _required_float(config: Mapping[str, Any], key: str) -> float:
    if key not in config:
        raise _InvalidValidityInput(f"missing configured threshold {key}")
    value = float(config[key])
    if not np.isfinite(value):
        raise _InvalidValidityInput(f"configured threshold {key} is not finite")
    return value


def _required_bool(config: Mapping[str, Any], key: str) -> bool:
    if key not in config or not isinstance(config[key], (bool, np.bool_)):
        raise _InvalidValidityInput(f"missing configured boolean {key}")
    return bool(config[key])


def _bounds(config: Mapping[str, Any], key: str) -> tuple[float, float]:
    if key not in config:
        raise _InvalidValidityInput(f"missing configured bounds {key}")
    values = config[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise _InvalidValidityInput(f"configured bounds {key} are not a pair")
    if len(values) != 2:
        raise _InvalidValidityInput(f"configured bounds {key} are not a pair")
    lower, upper = float(values[0]), float(values[1])
    if not np.isfinite([lower, upper]).all() or lower < 0.0 or lower > upper:
        raise _InvalidValidityInput(f"configured bounds {key} are invalid")
    return lower, upper


def _vector(payload: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise _InvalidValidityInput(f"missing numeric input {key}")
    try:
        values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise _InvalidValidityInput(f"numeric input {key} is invalid") from error
    if values.size == 0:
        raise _InvalidValidityInput(f"numeric input {key} is empty")
    return values


def _safe_quantiles(values: np.ndarray) -> dict[str, float | None]:
    names = ("q00", "q25", "q50", "q75", "q95", "q100")
    if not np.isfinite(values).all():
        return {name: None for name in names}
    quantiles = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {name: float(value) for name, value in zip(names, quantiles)}


def _waveform_diagnostics(
    input_values: np.ndarray,
    output_values: np.ndarray,
    channelwise_variance_ratio: np.ndarray,
) -> Result:
    if input_values.size != output_values.size:
        raise _InvalidValidityInput(
            "input_waveform_values and output_waveform_values must align"
        )
    if np.any(channelwise_variance_ratio < 0.0):
        raise _InvalidValidityInput(
            "channelwise_variance_ratio values must be nonnegative"
        )
    finite_waveforms = bool(
        np.isfinite(input_values).all() and np.isfinite(output_values).all()
    )
    if finite_waveforms:
        output_quantiles = np.quantile(output_values, (0.01, 0.50, 0.99))
        maximum_absolute = float(np.max(np.abs(output_values)))
        centered_input = input_values - np.mean(input_values)
        centered_output = output_values - np.mean(output_values)
        denominator = float(
            np.linalg.norm(centered_input) * np.linalg.norm(centered_output)
        )
        correlation = (
            None
            if denominator == 0.0
            else float(np.dot(centered_input, centered_output) / denominator)
        )
        waveform_quantiles: dict[str, float | None] = {
            "q0.01": float(output_quantiles[0]),
            "q0.50": float(output_quantiles[1]),
            "q0.99": float(output_quantiles[2]),
        }
    else:
        waveform_quantiles = {"q0.01": None, "q0.50": None, "q0.99": None}
        maximum_absolute = None
        correlation = None
    return {
        "output_waveform_quantiles": waveform_quantiles,
        "output_waveform_max_abs": maximum_absolute,
        "waveform_correlation": correlation,
        "channelwise_variance_ratio_distribution": {
            "count": int(channelwise_variance_ratio.size),
            "values": (
                [float(value) for value in channelwise_variance_ratio]
                if np.isfinite(channelwise_variance_ratio).all()
                else None
            ),
            "quantiles": _safe_quantiles(channelwise_variance_ratio),
        },
    }


def _check(observed: Any, comparator: str, threshold: Any, passed: bool) -> Result:
    return {
        "observed": observed,
        "comparator": comparator,
        "configured_threshold": threshold,
        "passed": bool(passed),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _finish(level: str, checks: Mapping[str, Result], **extra: Any) -> Result:
    passed = bool(checks) and all(bool(item["passed"]) for item in checks.values())
    return {
        "validity_level": level,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": dict(checks),
        **extra,
    }


def _blocked(level: str, error: Exception) -> Result:
    return {
        "validity_level": level,
        "status": "blocked_invalid_config_or_input",
        "passed": False,
        "reason": str(error),
        "checks": {},
    }


def _evaluate_v0(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    level = _level_config(config, "V0")
    finite_required = _required_bool(level, "all_finite")
    maximum_ratio = _required_float(
        level, "maximum_per_window_output_input_RMS_ratio"
    )
    full_bounds = _bounds(level, "full_median_output_input_RMS_ratio")
    low_bounds = _bounds(level, "low_artifact_median_output_input_RMS_ratio")
    maximum_low_change = _required_float(
        level, "low_artifact_maximum_median_relative_observation_change"
    )
    if maximum_ratio < 0.0 or maximum_low_change < 0.0:
        raise _InvalidValidityInput("configured V0 scale limits must be nonnegative")

    ratios = _vector(payload, "output_input_rms_ratio")
    low_ratios = _vector(payload, "low_artifact_output_input_rms_ratio")
    low_changes = _vector(
        payload, "low_artifact_relative_observation_change"
    )
    input_waveform = _vector(payload, "input_waveform_values")
    output_waveform = _vector(payload, "output_waveform_values")
    channelwise_variance_ratio = _vector(payload, "channelwise_variance_ratio")
    waveform_diagnostics = _waveform_diagnostics(
        input_waveform, output_waveform, channelwise_variance_ratio
    )
    if np.any(ratios < 0.0) or np.any(low_ratios < 0.0) or np.any(low_changes < 0.0):
        raise _InvalidValidityInput("V0 ratios and relative changes must be nonnegative")

    kind = str(payload.get("span_consistency_kind", ""))
    if kind == "complement":
        span_threshold_key = (
            "pure_operator_maximum_complement_consistency_relative_error"
        )
    elif kind == "union":
        span_threshold_key = "pure_operator_maximum_union_span_consistency_relative_error"
    else:
        raise _InvalidValidityInput(
            "span_consistency_kind must be complement or union"
        )
    span_threshold = _required_float(level, span_threshold_key)
    if span_threshold < 0.0:
        raise _InvalidValidityInput("configured span error limit must be nonnegative")
    span_error = float(payload.get("span_consistency_relative_error", np.nan))
    if span_error < 0.0:
        raise _InvalidValidityInput("span consistency relative error must be nonnegative")

    finite = bool(
        np.isfinite(ratios).all()
        and np.isfinite(low_ratios).all()
        and np.isfinite(low_changes).all()
        and np.isfinite(input_waveform).all()
        and np.isfinite(output_waveform).all()
        and np.isfinite(channelwise_variance_ratio).all()
        and np.isfinite(span_error)
    )
    full_quantiles = _safe_quantiles(ratios)
    low_quantiles = _safe_quantiles(low_ratios)
    change_quantiles = _safe_quantiles(low_changes)
    variance_quantiles = _safe_quantiles(np.square(ratios))
    low_variance_quantiles = _safe_quantiles(np.square(low_ratios))

    full_median = full_quantiles["q50"]
    low_median = low_quantiles["q50"]
    low_change_median = change_quantiles["q50"]
    maximum_observed = full_quantiles["q100"]
    full_variance_median = variance_quantiles["q50"]
    low_variance_median = low_variance_quantiles["q50"]
    checks = {
        "all_finite": _check(
            finite,
            "==" if finite_required else "not_required",
            True if finite_required else "not_required",
            not finite_required or finite,
        ),
        "per_window_scale_safety": _check(
            maximum_observed,
            "<=",
            maximum_ratio,
            maximum_observed is not None and maximum_observed <= maximum_ratio,
        ),
        "full_quantile_q50_bounds": _check(
            full_median,
            "within_inclusive",
            list(full_bounds),
            full_median is not None and full_bounds[0] <= full_median <= full_bounds[1],
        ),
        "low_artifact_quantile_q50_bounds": _check(
            low_median,
            "within_inclusive",
            list(low_bounds),
            low_median is not None and low_bounds[0] <= low_median <= low_bounds[1],
        ),
        "full_variance_ratio_q50_bounds": _check(
            full_variance_median,
            "within_inclusive",
            [full_bounds[0] ** 2, full_bounds[1] ** 2],
            full_variance_median is not None
            and full_bounds[0] ** 2 <= full_variance_median <= full_bounds[1] ** 2,
        ),
        "low_artifact_variance_ratio_q50_bounds": _check(
            low_variance_median,
            "within_inclusive",
            [low_bounds[0] ** 2, low_bounds[1] ** 2],
            low_variance_median is not None
            and low_bounds[0] ** 2
            <= low_variance_median
            <= low_bounds[1] ** 2,
        ),
        "low_artifact_observation_change": _check(
            low_change_median,
            "<=",
            maximum_low_change,
            low_change_median is not None and low_change_median <= maximum_low_change,
        ),
        f"{kind}_span_consistency": _check(
            span_error if np.isfinite(span_error) else None,
            "<=",
            span_threshold,
            np.isfinite(span_error) and span_error <= span_threshold,
        ),
    }
    return _finish(
        "V0",
        checks,
        metrics={
            "output_input_rms_ratio_quantiles": full_quantiles,
            "low_artifact_output_input_rms_ratio_quantiles": low_quantiles,
            "output_input_variance_ratio_quantiles": variance_quantiles,
            "low_artifact_output_input_variance_ratio_quantiles": (
                low_variance_quantiles
            ),
            "low_artifact_relative_change_quantiles": change_quantiles,
            "span_consistency_kind": kind,
            "span_consistency_relative_error": (
                span_error if np.isfinite(span_error) else None
            ),
            **waveform_diagnostics,
        },
    )


def evaluate_v0(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    """Evaluate scale/span gates and record valid-sample waveform diagnostics.

    ``input_waveform_values`` and ``output_waveform_values`` must be aligned,
    flattened valid samples; padding and guards must already be removed.
    ``channelwise_variance_ratio`` is likewise supplied after valid-time
    masking so this pure-numeric layer never guesses an array layout.
    """

    try:
        return _evaluate_v0(config, payload)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked("V0", error)


def _evaluate_v1(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    level = _level_config(config, "V1")
    minimum_reduction = _required_float(level, "minimum_relative_loss_reduction")
    maximum_rmse = _required_float(level, "maximum_standardized_latent_RMSE")
    maximum_identity_change = _required_float(
        level, "maximum_zero_artifact_relative_observation_change"
    )
    same_target_required = _required_bool(level, "both_models_must_fit_same_target")
    configured_timesteps = tuple(int(value) for value in level.get("timesteps", ()))
    if (
        not configured_timesteps
        or any(value < 0 for value in configured_timesteps)
        or len(set(configured_timesteps)) != len(configured_timesteps)
    ):
        raise _InvalidValidityInput("configured V1 timesteps are missing or duplicated")
    if not 0.0 <= minimum_reduction <= 1.0 or min(
        maximum_rmse, maximum_identity_change
    ) < 0.0:
        raise _InvalidValidityInput("configured V1 numeric limits are invalid")

    raw_rows = payload.get("timestep_results")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise _InvalidValidityInput("timestep_results must be a nonempty sequence")
    if not raw_rows:
        raise _InvalidValidityInput("timestep_results must be a nonempty sequence")
    rows: list[Result] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise _InvalidValidityInput("each V1 timestep result must be a mapping")
        model_id = str(raw.get("model_id", ""))
        target_id = str(raw.get("target_id", ""))
        timestep = int(raw["timestep"])
        key = (model_id, timestep)
        if not model_id or not target_id or key in seen:
            raise _InvalidValidityInput("V1 model/target IDs are missing or duplicated")
        seen.add(key)
        initial_loss = float(raw["initial_loss"])
        final_loss = float(raw["final_loss"])
        latent_rmse = float(raw["standardized_latent_rmse"])
        identity_change = float(raw["zero_artifact_relative_observation_change"])
        values = (initial_loss, final_loss, latent_rmse, identity_change)
        if not np.isfinite(values).all() or initial_loss <= 0.0 or any(
            value < 0.0 for value in values[1:]
        ):
            raise _InvalidValidityInput("V1 numeric results are invalid")
        reduction = (initial_loss - final_loss) / initial_loss
        row_passed = bool(
            reduction >= minimum_reduction
            and latent_rmse <= maximum_rmse
            and identity_change <= maximum_identity_change
        )
        rows.append(
            {
                "model_id": model_id,
                "target_id": target_id,
                "timestep": timestep,
                "relative_loss_reduction": float(reduction),
                "standardized_latent_rmse": latent_rmse,
                "zero_artifact_relative_observation_change": identity_change,
                "passed": row_passed,
            }
        )

    model_ids = sorted({str(row["model_id"]) for row in rows})
    target_ids = sorted({str(row["target_id"]) for row in rows})
    timestep_coverage = {
        model_id: sorted(
            int(row["timestep"])
            for row in rows
            if row["model_id"] == model_id
        )
        for model_id in model_ids
    }
    coverage_ok = bool(
        model_ids
        and all(
            tuple(timestep_coverage[model_id]) == tuple(sorted(configured_timesteps))
            for model_id in model_ids
        )
    )
    same_target_ok = bool(
        not same_target_required
        or (len(model_ids) == 2 and len(target_ids) == 1)
    )
    checks = {
        "configured_timestep_coverage": _check(
            timestep_coverage,
            "exact_per_model",
            list(sorted(configured_timesteps)),
            coverage_ok,
        ),
        "both_models_same_target": _check(
            {"models": model_ids, "targets": target_ids},
            "two_models_one_target" if same_target_required else "not_required",
            same_target_required,
            same_target_ok,
        ),
        "minimum_relative_loss_reduction": _check(
            min((float(row["relative_loss_reduction"]) for row in rows), default=None),
            ">=",
            minimum_reduction,
            bool(rows) and all(
                float(row["relative_loss_reduction"]) >= minimum_reduction
                for row in rows
            ),
        ),
        "maximum_standardized_latent_rmse": _check(
            max((float(row["standardized_latent_rmse"]) for row in rows), default=None),
            "<=",
            maximum_rmse,
            bool(rows) and all(
                float(row["standardized_latent_rmse"]) <= maximum_rmse
                for row in rows
            ),
        ),
        "zero_artifact_identity": _check(
            max(
                (
                    float(row["zero_artifact_relative_observation_change"])
                    for row in rows
                ),
                default=None,
            ),
            "<=",
            maximum_identity_change,
            bool(rows) and all(
                float(row["zero_artifact_relative_observation_change"])
                <= maximum_identity_change
                for row in rows
            ),
        ),
    }
    return _finish("V1", checks, timestep_results=rows)


def evaluate_v1(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    """Evaluate target fitting and zero-artifact identity at frozen timesteps."""

    try:
        return _evaluate_v1(config, payload)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked("V1", error)


def _growth_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _evaluate_v2(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    level = _level_config(config, "V2")
    maximum_growth = _required_float(
        level, "maximum_unexplained_adjacent_RMS_ratio"
    )
    exponential_forbidden = _required_bool(level, "exponential_growth_forbidden")
    final_v0_required = _required_bool(level, "final_output_must_pass_V0")
    if maximum_growth < 1.0:
        raise _InvalidValidityInput("V2 maximum RMS growth ratio must be at least one")

    raw_trajectories = payload.get("trajectories")
    if not isinstance(raw_trajectories, Sequence) or isinstance(
        raw_trajectories, (str, bytes)
    ):
        raise _InvalidValidityInput("trajectories must be a nonempty sequence")
    if not raw_trajectories:
        raise _InvalidValidityInput("trajectories must be a nonempty sequence")
    summaries: list[Result] = []
    chart_data: list[Result] = []
    for raw in raw_trajectories:
        if not isinstance(raw, Mapping):
            raise _InvalidValidityInput("each V2 trajectory must be a mapping")
        trajectory_id = str(raw.get("trajectory_id", ""))
        rms = np.asarray(raw.get("rms", ()), dtype=np.float64).reshape(-1)
        if not trajectory_id or rms.size < 2:
            raise _InvalidValidityInput("V2 trajectories need an ID and two RMS states")
        if not np.isfinite(rms).all() or np.any(rms < 0.0):
            raise _InvalidValidityInput("V2 RMS trajectories must be finite and nonnegative")
        raw_steps = raw.get("steps", list(range(rms.size)))
        steps = np.asarray(raw_steps, dtype=np.int64).reshape(-1)
        if steps.size != rms.size or len(set(int(value) for value in steps)) != rms.size:
            raise _InvalidValidityInput("V2 trajectory steps are missing or duplicated")

        adjacent = [1.0]
        prior_minimum = float(rms[0])
        growth_from_prior_minimum = [1.0]
        for index in range(1, rms.size):
            adjacent.append(_growth_ratio(float(rms[index]), float(rms[index - 1])))
            growth_from_prior_minimum.append(
                _growth_ratio(float(rms[index]), prior_minimum)
            )
            prior_minimum = min(prior_minimum, float(rms[index]))
        maximum_adjacent = max(adjacent)
        maximum_cumulative = max(growth_from_prior_minimum)
        jump_passed = maximum_adjacent <= maximum_growth
        exponential_detected = maximum_cumulative > maximum_growth
        exponential_passed = not exponential_forbidden or not exponential_detected
        summaries.append(
            {
                "trajectory_id": trajectory_id,
                "maximum_adjacent_growth_ratio": _finite_or_none(maximum_adjacent),
                "maximum_growth_from_prior_minimum": _finite_or_none(
                    maximum_cumulative
                ),
                "unexplained_jump_passed": bool(jump_passed),
                "exponential_growth_detected": bool(exponential_detected),
                "passed": bool(jump_passed and exponential_passed),
            }
        )
        chart_data.extend(
            {
                "trajectory_id": trajectory_id,
                "step": int(step),
                "rms": float(value),
                "adjacent_growth_ratio": _finite_or_none(adjacent[index]),
                "growth_from_prior_minimum": _finite_or_none(
                    growth_from_prior_minimum[index]
                ),
            }
            for index, (step, value) in enumerate(zip(steps, rms))
        )

    final_v0_status = str(payload.get("final_v0_status", "missing"))
    maximum_adjacent_observed = (
        None
        if any(
            item["maximum_adjacent_growth_ratio"] is None for item in summaries
        )
        else max(
            float(item["maximum_adjacent_growth_ratio"])
            for item in summaries
        )
    )
    checks = {
        "no_unexplained_adjacent_jump": _check(
            maximum_adjacent_observed,
            "<=",
            maximum_growth,
            bool(summaries)
            and all(bool(item["unexplained_jump_passed"]) for item in summaries),
        ),
        "no_exponential_expansion": _check(
            any(bool(item["exponential_growth_detected"]) for item in summaries),
            "==",
            False if exponential_forbidden else "not_required",
            bool(summaries)
            and all(
                not exponential_forbidden
                or not bool(item["exponential_growth_detected"])
                for item in summaries
            ),
        ),
        "final_output_v0": _check(
            final_v0_status,
            "==",
            "passed" if final_v0_required else "not_required",
            not final_v0_required or final_v0_status == "passed",
        ),
    }
    return _finish(
        "V2", checks, trajectory_summaries=summaries, chart_data=chart_data
    )


def evaluate_v2(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    """Evaluate trajectory stability and return plot-ready numeric rows."""

    try:
        return _evaluate_v2(config, payload)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked("V2", error)


def _numeric_context_mapping(
    payload: Mapping[str, Any], key: str, required: set[str]
) -> dict[str, float]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise _InvalidValidityInput(
            f"{key} must contain exactly {sorted(required)}"
        )
    values = {str(name): float(value) for name, value in raw.items()}
    if not np.isfinite(list(values.values())).all() or any(
        value < 0.0 for value in values.values()
    ):
        raise _InvalidValidityInput(f"{key} values must be finite and nonnegative")
    return values


def _evaluate_v3(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    level = _level_config(config, "V3")
    maximum_repeat = _required_float(level, "maximum_repeat_relative_difference")
    minimum_context_change = _required_float(
        level, "minimum_context_swap_artifact_relative_change"
    )
    if maximum_repeat < 0.0 or minimum_context_change < 0.0:
        raise _InvalidValidityInput("configured V3 numeric limits must be nonnegative")
    all_contexts_required = _required_bool(
        level, "matching_population_wrong_shuffled_required"
    )
    wrong_shuffled_safety_required = _required_bool(
        level, "wrong_and_shuffled_scale_safety_required"
    )
    rho_fixed_required = _required_bool(level, "original_subject_rho_held_fixed")
    required_contexts = {"population", "matching", "wrong", "shuffled"}
    swapped_contexts = {"population", "wrong", "shuffled"}

    repeats = _numeric_context_mapping(
        payload, "repeat_relative_difference_by_context", required_contexts
    )
    changes = _numeric_context_mapping(
        payload, "context_swap_artifact_relative_change", swapped_contexts
    )
    raw_safety = payload.get("scale_safety_by_context")
    if not isinstance(raw_safety, Mapping) or set(raw_safety) != required_contexts:
        raise _InvalidValidityInput(
            "scale_safety_by_context must contain population/matching/wrong/shuffled"
        )
    if not all(isinstance(value, (bool, np.bool_)) for value in raw_safety.values()):
        raise _InvalidValidityInput("scale safety values must be booleans")
    safety = {str(name): bool(value) for name, value in raw_safety.items()}
    rho = _numeric_context_mapping(payload, "rho_by_context", required_contexts)
    if any(value > 1.0 for value in rho.values()):
        raise _InvalidValidityInput("rho_by_context values must lie in [0,1]")

    context_presence_ok = not all_contexts_required or bool(
        set(repeats) == required_contexts
        and set(safety) == required_contexts
        and set(rho) == required_contexts
    )
    rho_values = list(rho.values())
    rho_fixed = len(set(rho_values)) == 1
    checks = {
        "repeatability": _check(
            max(repeats.values()), "<=", maximum_repeat, max(repeats.values()) <= maximum_repeat
        ),
        "context_swap_changes_artifact": _check(
            min(changes.values()),
            ">=",
            minimum_context_change,
            min(changes.values()) >= minimum_context_change,
        ),
        "required_contexts_present": _check(
            sorted(required_contexts),
            "present",
            all_contexts_required,
            context_presence_ok,
        ),
        "wrong_and_shuffled_scale_safety": _check(
            {name: safety[name] for name in ("wrong", "shuffled")},
            "all_true" if wrong_shuffled_safety_required else "not_required",
            wrong_shuffled_safety_required,
            not wrong_shuffled_safety_required
            or (safety["wrong"] and safety["shuffled"]),
        ),
        "original_subject_rho_fixed": _check(
            rho,
            "all_exactly_equal" if rho_fixed_required else "not_required",
            rho_fixed_required,
            not rho_fixed_required or rho_fixed,
        ),
    }
    return _finish(
        "V3",
        checks,
        metrics={
            "repeat_relative_difference_by_context": repeats,
            "context_swap_artifact_relative_change": changes,
            "scale_safety_by_context": safety,
            "rho_by_context": rho,
        },
    )


def evaluate_v3(config: Mapping[str, Any], payload: Mapping[str, Any]) -> Result:
    """Evaluate repeatability, context response, scale safety, and fixed rho."""

    try:
        return _evaluate_v3(config, payload)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked("V3", error)


def evaluate_validity(
    level: str, config: Mapping[str, Any], payload: Mapping[str, Any]
) -> Result:
    """Dispatch one configured validity level without performing any I/O."""

    evaluators: dict[str, Evaluator] = {
        "V0": evaluate_v0,
        "V1": evaluate_v1,
        "V2": evaluate_v2,
        "V3": evaluate_v3,
    }
    if level not in evaluators:
        return _blocked(level, _InvalidValidityInput(f"unknown validity level {level}"))
    return evaluators[level](config, payload)


__all__ = [
    "evaluate_v0",
    "evaluate_v1",
    "evaluate_v2",
    "evaluate_v3",
    "evaluate_validity",
]
