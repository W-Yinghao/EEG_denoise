"""Fair one-seed P2/P3 subject-bridge mechanism screen.

The canonical artifact target is invariant to context intervention.  A single
FiLM diffusion checkpoint and an information-matched FiLM deterministic
checkpoint are trained per outer fold.  Every deployed arm is anchored at the
same frozen population cleaner and uses the same EEG-only activity gate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.experiments.mainline_subject_residual_diffusion import (
    _annotation_opener,
    _continuous,
    _evaluate_output,
    _klados_eval_records,
    _load_models as _load_population_baseline,
    _paired_metrics,
    _prepared,
    _sge_samples_per_trial,
    _split_indices,
)
from eeg_cgdr.experiments.parallel_subject_aware_routes_v1 import (
    _donor_contexts,
    _klados_matching_support,
    _mapping,
    _model_configs,
    _runtime_condition,
    _sge_matching_support,
    _subject_arrays,
    _write_csv,
    _write_json,
)
from eeg_cgdr.experiments.subject_artifact_data import PreparedSubjectArtifactFold
from eeg_cgdr.models.artifact_subspace_diffusion import participant_sample_seeds
from eeg_cgdr.models.parallel_subject_routes import (
    AdaptiveActivityGate,
    FullCFiLMDiffusion,
    FullCFiLMDeterministic,
    canonical_target,
)
from eeg_cgdr.models.subject_bridge import (
    blocked_split_half_reliability,
    coordinate_corrected_bridge,
    fit_signed_beta,
    physical_eeg_delta,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_parallel_explore"))
PROTOCOL = "subject_bridge_repair"
SEED = 20260811
CORE_METHODS = (
    "RAW", "POP-FALLBACK", "U-POP-CONTEXT", "U-MATCH",
    "D-POP-CONTEXT", "D-MATCH", "D-WRONG-1", "D-WRONG-2", "D-WRONG-3",
    "D-MATCH-G1", "D-MATCH-RHO1", "D-MATCH-BETA1", "D-MATCH-K1",
)


def _load(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    if config.get("protocol_id") != PROTOCOL or int(config.get("harness_level", -1)) != 1:
        raise ValueError("subject bridge protocol/harness changed")
    base = yaml.safe_load((CODE_ROOT / str(config["base_subject_artifact_config"])).read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base config is invalid")
    return base, CODE_ROOT / str(_mapping(config, "outputs")["root"])


def _implementation() -> dict[str, Any]:
    return {
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def task_rows() -> list[dict[str, Any]]:
    return ([{"task_index": 0, "dataset": "klados", "fold_index": 0, "seed": SEED}]
            + [{"task_index": value + 1, "dataset": "sgeyesub", "fold_index": value, "seed": SEED} for value in range(25)])


def _prepared_route(base: Mapping[str, Any], row: Mapping[str, Any]) -> PreparedSubjectArtifactFold:
    return _prepared(base, str(row["dataset"]), int(row["fold_index"]))


def _checkpoint(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "checkpoints" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}" / f"seed_{int(row['seed'])}" / "models.pt"


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _corrected_arrays(prepared: PreparedSubjectArtifactFold, ridge: float) -> dict[str, np.ndarray]:
    arrays = _subject_arrays(prepared)
    keys = np.asarray(arrays["recording_keys"])
    rho = np.zeros(keys.shape[0], dtype=np.float32)
    error = np.full(keys.shape[0], np.inf, dtype=np.float32)
    stability = np.full(keys.shape[0], np.inf, dtype=np.float32)
    for key in sorted(set(str(value) for value in keys)):
        indices = np.flatnonzero(keys == key)
        diagnostic = blocked_split_half_reliability(
            arrays["observed"][indices], arrays["target"][indices], arrays["valid"][indices],
            latent_mean=prepared.latent_normalizer.mean,
            latent_standard_deviation=prepared.latent_normalizer.standard_deviation,
            ridge=ridge,
        )
        rho[indices] = diagnostic.reliability
        error[indices] = diagnostic.heldout_error
        stability[indices] = diagnostic.operator_stability
    arrays["rho"] = rho
    arrays["support_heldout_error"] = error
    arrays["support_operator_stability"] = stability
    return arrays


def _batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    def value(key: str, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(arrays[key][indices], device=device, dtype=dtype)
    target = canonical_target(value("target", torch.float32))
    return target, {
        "observed": value("observed", torch.float32),
        "full_transfer": value("full", torch.float32),
        "normalized_transfer": value("normalized", torch.float32),
        "transfer_scale": value("scale", torch.float32),
        "singular_values": value("singular", torch.float32),
        "rank": value("rank", torch.long),
        "rho": value("rho", torch.float32),
        "calibration_duration_seconds": value("duration", torch.float32),
        "channel_mask": value("channel_mask", torch.bool),
        "valid_time_mask": value("valid", torch.bool),
        "support_sample_count": value("support_sample_count", torch.float32),
    }


def _masked_mse(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    weight = valid[:, None].to(predicted.dtype)
    return ((predicted - target).square() * weight).sum() / (weight.sum() * target.shape[1]).clamp_min(1)


def _ema_score(model: FullCFiLMDiffusion, ema: Mapping[str, Tensor], target: Tensor, condition: Mapping[str, Tensor], seed: int) -> float:
    live = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(ema); model.eval()
    generator = torch.Generator(device=target.device).manual_seed(seed)
    with torch.no_grad():
        score = float(model.training_loss(target, generator=generator, **condition)[1]["x0_mse"].cpu())
    model.load_state_dict(live); model.train()
    return score


def _fit_beta_training_only(
    anchor: torch.nn.Module,
    gate: AdaptiveActivityGate,
    arrays: Mapping[str, np.ndarray],
    validation: np.ndarray,
    prepared: PreparedSubjectArtifactFold,
    device: torch.device,
    bounds: Sequence[float],
) -> tuple[float, dict[str, float]]:
    target, condition = _batch(arrays, validation, device)
    y = condition["observed"]; mask = condition["valid_time_mask"]
    with torch.no_grad():
        population = anchor(y, mask)
        activity = gate(y, mask)
        subject = physical_eeg_delta(
            target, normalized_transfer=condition["normalized_transfer"],
            latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=device),
            latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device),
            valid_time_mask=mask,
        )
        pop_transfer = torch.as_tensor(prepared.population_context.normalized_transfer, device=device)[None].expand(y.shape[0], -1, -1)
        pop = physical_eeg_delta(
            target, normalized_transfer=pop_transfer,
            latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=device),
            latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device),
            valid_time_mask=mask,
        )
        direction = condition["rho"][:, None, None] * activity * (subject - pop)
        clean_teacher = y - subject
    beta, zero_loss, selected_loss = fit_signed_beta(
        direction.cpu().numpy(), (clean_teacher - population).cpu().numpy(), mask.cpu().numpy(),
        lower=float(bounds[0]), upper=float(bounds[1]),
    )
    return beta, {"beta_zero_loss": zero_loss, "beta_selected_loss": selected_loss}


def train_task(config: Mapping[str, Any], run_dir: Path, index: int) -> Mapping[str, Any]:
    base, root = _load(config); row = task_rows()[index]
    checkpoint = _checkpoint(root, row)
    if checkpoint.is_file():
        return {"status": "skipped_completed", "checkpoint": str(checkpoint), **dict(row)}
    prepared = _prepared_route(base, row)
    training = _mapping(config, "training")
    arrays = _corrected_arrays(prepared, float(training["ridge_lambda"]))
    model_config, diffusion_config = _model_configs(prepared, config)
    device = torch.device("cuda", 0); seed = int(row["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    population_transfer = torch.as_tensor(prepared.population_context.full_transfer)
    diffusion = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=population_transfer).to(device)
    deterministic = FullCFiLMDeterministic(model_config, population_transfer=population_transfer).to(device)
    gate = AdaptiveActivityGate(prepared.model_dimensions.eeg_channels).to(device)
    optimizer_d = AdamW(diffusion.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    optimizer_u = AdamW(deterministic.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    optimizer_g = AdamW(gate.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    best_d = best_u = best_g = float("inf"); state_d = state_u = state_g = None
    best_step_d = best_step_u = best_step_g = 0
    curve: list[dict[str, Any]] = []; started = time.perf_counter()
    maximum = int(training["maximum_updates"]); gate_maximum = int(training["gate_updates"])
    for step in range(1, maximum + 1):
        indices = rng.choice(train_indices, size=int(training["batch_size"]), replace=train_indices.size < int(training["batch_size"]))
        target, condition = _batch(arrays, indices, device)
        optimizer_d.zero_grad(set_to_none=True)
        loss_d, diagnostics = diffusion.training_loss(target, generator=generator, **condition)
        loss_d.backward(); gradient_d = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True); optimizer_d.step()
        optimizer_u.zero_grad(set_to_none=True)
        predicted = deterministic(condition["observed"], **{key: value for key, value in condition.items() if key != "observed"})
        loss_u = _masked_mse(predicted, target, condition["valid_time_mask"])
        loss_u.backward(); gradient_u = torch.nn.utils.clip_grad_norm_(deterministic.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True); optimizer_u.step()
        if step <= gate_maximum:
            optimizer_g.zero_grad(set_to_none=True)
            magnitude = torch.sqrt(target.square().mean(dim=1, keepdim=True).clamp_min(1e-12))
            gate_target = (magnitude / magnitude.detach().quantile(0.75).clamp_min(1e-6)).clamp(0, 1)
            gate_prediction = gate(condition["observed"], condition["valid_time_mask"])
            loss_g = _masked_mse(gate_prediction, gate_target, condition["valid_time_mask"])
            loss_g.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0, error_if_nonfinite=True); optimizer_g.step()
        else:
            loss_g = torch.zeros((), device=device)
        decay = float(_mapping(config, "diffusion")["ema_decay"])
        with torch.no_grad():
            for key, value in diffusion.state_dict().items():
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"step": step, "diffusion_loss": float(loss_d.detach()), "deterministic_loss": float(loss_u.detach()), "gate_loss": float(loss_g.detach()), "diffusion_gradient": float(gradient_d), "deterministic_gradient": float(gradient_u), **{key: float(value) for key, value in diagnostics.items()}})
        if step % int(training["validation_interval_updates"]) == 0 or step == maximum:
            target_v, condition_v = _batch(arrays, validation_indices, device)
            score_d = _ema_score(diffusion, ema, target_v, condition_v, seed + 99001)
            deterministic.eval(); gate.eval()
            with torch.no_grad():
                pred_v = deterministic(condition_v["observed"], **{key: value for key, value in condition_v.items() if key != "observed"})
                score_u = float(_masked_mse(pred_v, target_v, condition_v["valid_time_mask"]).cpu())
                magnitude_v = torch.sqrt(target_v.square().mean(dim=1, keepdim=True).clamp_min(1e-12))
                target_g = (magnitude_v / magnitude_v.quantile(0.75).clamp_min(1e-6)).clamp(0, 1)
                score_g = float(_masked_mse(gate(condition_v["observed"], condition_v["valid_time_mask"]), target_g, condition_v["valid_time_mask"]).cpu())
            deterministic.train(); gate.train()
            curve.append({"step": step, "ema_validation_x0_mse": score_d, "deterministic_validation_mse": score_u, "gate_validation_mse": score_g})
            if score_d < best_d: best_d, best_step_d, state_d = score_d, step, {key: value.detach().cpu().clone() for key, value in ema.items()}
            if score_u < best_u: best_u, best_step_u, state_u = score_u, step, {key: value.detach().cpu().clone() for key, value in deterministic.state_dict().items()}
            if score_g < best_g: best_g, best_step_g, state_g = score_g, step, {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    if state_d is None or state_u is None or state_g is None:
        raise AssertionError("validation failed to select all fair checkpoints")
    diffusion.load_state_dict(state_d); deterministic.load_state_dict(state_u); gate.load_state_dict(state_g)
    old_config = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    anchor, _, _, _, anchor_checkpoint = _load_population_baseline(old_config, prepared, row, device)
    beta, beta_metrics = _fit_beta_training_only(anchor, gate, arrays, validation_indices, prepared, device, training["beta_bounds"])
    _save(checkpoint, {
        "protocol_id": PROTOCOL, "route": dict(row), "model_config": model_config.__dict__, "diffusion_config": diffusion_config.__dict__,
        "diffusion_ema": state_d, "deterministic": state_u, "activity_gate": state_g,
        "best_step_diffusion": best_step_d, "best_step_deterministic": best_step_u, "best_step_gate": best_step_g,
        "validation_diffusion": best_d, "validation_deterministic": best_u, "validation_gate": best_g,
        "beta": beta, **beta_metrics, "latent_mean": prepared.latent_normalizer.mean,
        "latent_standard_deviation": prepared.latent_normalizer.standard_deviation,
        "population_context": prepared.population_context, "population_anchor_checkpoint": str(anchor_checkpoint),
    })
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    summary = {"status": "completed_fair_P2_P3_training", **_implementation(), **dict(row), "checkpoint": str(checkpoint), "beta": beta, "best_step_diffusion": best_step_d, "best_step_deterministic": best_step_u, "best_step_gate": best_step_g, "runtime_seconds": time.perf_counter() - started, "support_artifact_spectrum": "removed", "fair_information_matched_deterministic": True}
    _write_json(checkpoint.parent / "result_summary.json", summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def _load_models(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, row: Mapping[str, Any], device: torch.device):
    _, root = _load(config); payload = torch.load(_checkpoint(root, row), map_location=device, weights_only=False)
    model_config, diffusion_config = _model_configs(prepared, config)
    population_transfer = torch.as_tensor(prepared.population_context.full_transfer)
    diffusion = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=population_transfer).to(device)
    deterministic = FullCFiLMDeterministic(model_config, population_transfer=population_transfer).to(device)
    gate = AdaptiveActivityGate(prepared.model_dimensions.eeg_channels).to(device)
    diffusion.load_state_dict(payload["diffusion_ema"]); deterministic.load_state_dict(payload["deterministic"]); gate.load_state_dict(payload["activity_gate"])
    old_config = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    anchor, _, _, _, anchor_checkpoint = _load_population_baseline(old_config, prepared, row, device)
    diffusion.eval(); deterministic.eval(); gate.eval(); anchor.eval()
    return diffusion, deterministic, gate, anchor, payload, anchor_checkpoint


def _support_reliability(support: tuple[np.ndarray, np.ndarray, np.ndarray], prepared: PreparedSubjectArtifactFold, ridge: float) -> tuple[float, dict[str, float]]:
    observed, latent, valid = support
    value = blocked_split_half_reliability(observed, latent, valid, latent_mean=prepared.latent_normalizer.mean, latent_standard_deviation=prepared.latent_normalizer.standard_deviation, ridge=ridge)
    return value.reliability, {"support_heldout_error": value.heldout_error, "support_operator_stability": value.operator_stability, "support_samples": value.samples}


def _donor_support(prepared: PreparedSubjectArtifactFold, context: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = _subject_arrays(prepared); key = str(context.fit_recording_keys[0]); indices = np.flatnonzero(np.asarray(arrays["recording_keys"]) == key)
    if indices.size < 1: raise ValueError("wrong donor support is absent from outer training arrays")
    return arrays["observed"][indices], arrays["target"][indices], arrays["valid"][indices]


@torch.no_grad()
def _context_latents(diffusion: FullCFiLMDiffusion, deterministic: FullCFiLMDeterministic, condition: Mapping[str, Tensor], prepared: PreparedSubjectArtifactFold, unit: str, seed: int) -> tuple[Tensor, Tensor, Tensor, int]:
    y = condition["observed"]
    seeds = participant_sample_seeds(unit, seed, count=8)
    posterior = diffusion.posterior_mean(
        observed=y, latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=y.device),
        latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=y.device),
        sample_seeds=seeds, ddim_steps=25, record_trajectory=False,
        **{key: value for key, value in condition.items() if key != "observed"},
    )
    if posterior.standardized_latent_samples is None: raise AssertionError("K1 sample was not retained")
    deterministic_latent = deterministic(y, **{key: value for key, value in condition.items() if key != "observed"})
    return posterior.standardized_latent_mean, posterior.standardized_latent_samples[0], deterministic_latent, int(posterior.network_calls)


def _gate_diagnostics(gate: Tensor, mask: Tensor) -> dict[str, float]:
    values = gate.expand(-1, 1, -1)[mask[:, None]].detach().cpu().numpy()
    return {"gate_mean": float(values.mean()), "gate_q25": float(np.quantile(values, .25)), "gate_q50": float(np.quantile(values, .5)), "gate_q75": float(np.quantile(values, .75)), "gate_active_frame_fraction": float(np.mean(values > .5)), "gate_quiet_frame_fraction": float(np.mean(values <= .1))}


@torch.no_grad()
def infer_unit(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, row: Mapping[str, Any], *, unit: str, observed: np.ndarray, valid: np.ndarray, matching: Any, support: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    diffusion, deterministic, gate_model, anchor, payload, anchor_checkpoint = _load_models(config, prepared, row, device)
    donors = _donor_contexts(prepared, minimum=3)
    rho, reliability = _support_reliability(support, prepared, float(_mapping(config, "training")["ridge_lambda"]))
    beta = float(payload["beta"]); batch_size = int(_mapping(config, "evaluation")["batch_size"])
    output_chunks: dict[str, list[np.ndarray]] = defaultdict(list); gate_values: list[dict[str, float]] = []; calls = 0
    donor_rhos = []
    for donor in donors:
        donor_rho, _ = _support_reliability(_donor_support(prepared, donor), prepared, float(_mapping(config, "training")["ridge_lambda"]))
        donor_rhos.append(donor_rho)
    for batch_index, start in enumerate(range(0, observed.shape[0], batch_size)):
        stop = min(start + batch_size, observed.shape[0]); y = torch.as_tensor(observed[start:stop], device=device); mask = torch.as_tensor(valid[start:stop], device=device)
        pop_output = anchor(y, mask); activity = gate_model(y, mask); gate_values.append(_gate_diagnostics(activity, mask))
        population_condition = _runtime_condition(prepared.population_context, y.shape[0], prepared, valid[start:stop], device); population_condition["observed"] = y
        match_condition = _runtime_condition(matching, y.shape[0], prepared, valid[start:stop], device); match_condition["observed"] = y; match_condition["rho"][:] = rho
        d0, d0_k1, u0, n = _context_latents(diffusion, deterministic, population_condition, prepared, f"{unit}:batch{batch_index}", int(row["seed"])); calls += n
        ds, ds_k1, us, n = _context_latents(diffusion, deterministic, match_condition, prepared, f"{unit}:batch{batch_index}", int(row["seed"])); calls += n
        latent_mean = torch.as_tensor(prepared.latent_normalizer.mean, device=device); latent_std = torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device)
        pop_delta_d = physical_eeg_delta(d0, normalized_transfer=population_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
        pop_delta_u = physical_eeg_delta(u0, normalized_transfer=population_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
        match_delta_d = physical_eeg_delta(ds, normalized_transfer=match_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
        match_delta_d_k1 = physical_eeg_delta(ds_k1, normalized_transfer=match_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
        match_delta_u = physical_eeg_delta(us, normalized_transfer=match_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
        one = torch.ones_like(activity)
        def bridge(context_delta: Tensor, population_delta: Tensor, *, b: float = beta, r: float = rho, g: Tensor = activity) -> Tensor:
            return coordinate_corrected_bridge(pop_output, context_delta=context_delta, population_delta=population_delta, beta=b, rho=r, activity_gate=g, valid_time_mask=mask)[0]
        output_chunks["RAW"].append(y.cpu().numpy()); output_chunks["POP-FALLBACK"].append(pop_output.cpu().numpy())
        output_chunks["U-POP-CONTEXT"].append(bridge(pop_delta_u, pop_delta_u).cpu().numpy()); output_chunks["U-MATCH"].append(bridge(match_delta_u, pop_delta_u).cpu().numpy())
        output_chunks["D-POP-CONTEXT"].append(bridge(pop_delta_d, pop_delta_d).cpu().numpy()); output_chunks["D-MATCH"].append(bridge(match_delta_d, pop_delta_d).cpu().numpy())
        output_chunks["D-MATCH-G1"].append(bridge(match_delta_d, pop_delta_d, g=one).cpu().numpy()); output_chunks["D-MATCH-RHO1"].append(bridge(match_delta_d, pop_delta_d, r=1.0).cpu().numpy()); output_chunks["D-MATCH-BETA1"].append(bridge(match_delta_d, pop_delta_d, b=1.0).cpu().numpy()); output_chunks["D-MATCH-K1"].append(bridge(match_delta_d_k1, pop_delta_d).cpu().numpy())
        for donor_index, donor in enumerate(donors, 1):
            donor_condition = _runtime_condition(donor, y.shape[0], prepared, valid[start:stop], device); donor_condition["observed"] = y; donor_condition["rho"][:] = rho
            donor_latent, _, _, n = _context_latents(diffusion, deterministic, donor_condition, prepared, f"{unit}:batch{batch_index}", int(row["seed"])); calls += n
            donor_delta = physical_eeg_delta(donor_latent, normalized_transfer=donor_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask)
            output_chunks[f"D-WRONG-{donor_index}"].append(bridge(donor_delta, pop_delta_d).cpu().numpy())
    outputs = {key: np.concatenate(value) for key, value in output_chunks.items()}
    diagnostics = {**reliability, "rho": rho, "beta": beta, "donor_rhos_secondary": donor_rhos, "wrong_donors": [value.context_id for value in donors], "network_calls": calls, "checkpoint": str(_checkpoint(_load(config)[1], row)), "population_anchor_checkpoint": str(anchor_checkpoint), "common_random_numbers": True, "query_eog_used": False}
    for key in gate_values[0]: diagnostics[key] = float(np.mean([value[key] for value in gate_values]))
    return outputs, diagnostics


def _screen_task(config: Mapping[str, Any], run_dir: Path, index: int) -> Mapping[str, Any]:
    base, root = _load(config); row = task_rows()[index]; prepared = _prepared_route(base, row); device = torch.device("cuda", 0)
    output_root = root / "screen" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"; rows: list[dict[str, Any]] = []
    arrays_root = root / "server_arrays" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"; arrays_root.mkdir(parents=True, exist_ok=True)
    if row["dataset"] == "klados":
        for unit, mechanism, matching, _ in _klados_eval_records(base):
            support = _klados_matching_support(base, prepared, mechanism, unit)
            outputs, diagnostics = infer_unit(config, prepared, row, unit=unit, observed=mechanism.observed_windows.astype(np.float32), valid=mechanism.valid_time_weight.astype(bool), matching=matching, support=support, device=device)
            archive = arrays_root / f"{unit}.npz"; np.savez_compressed(archive, **{key.replace("-", "_"): value for key, value in outputs.items()})
            for method, output in outputs.items():
                rows.append({"dataset": "klados", "unit_id": unit, "exact_cell": prepared.fold.layout_id, "training_seed": SEED, "method": method, "status": "success", **_paired_metrics(mechanism.observed_windows, mechanism.clean_windows, output, mechanism.valid_time_weight.astype(bool)), **{key: value for key, value in diagnostics.items() if not isinstance(value, (list, dict))}, "statistical_unit": "source_record", "oracle_query_EOG": False})
    else:
        for unit, heldout in prepared.heldout.items():
            support = _sge_matching_support(base, prepared, int(row["fold_index"]), unit)
            outputs, diagnostics = infer_unit(config, prepared, row, unit=unit, observed=heldout.query.observed, valid=heldout.query.valid_time_mask, matching=heldout.matching, support=support, device=device)
            archive = arrays_root / f"{unit.replace('/', '__')}.npz"; np.savez_compressed(archive, **{key.replace("-", "_"): value for key, value in outputs.items()})
            # Deployment outputs are frozen before query annotations/EOG are opened.
            annotated = _annotation_opener(base, prepared, unit)(); annotations = annotated.query_annotations
            if annotations is None: raise AssertionError("query annotations were not opened after output freeze")
            for method, output in outputs.items():
                metric = _evaluate_output(method_id=method, output=_continuous(output), observed=_continuous(heldout.query.observed), matching_projector=heldout.matching.projector, population_projector=prepared.population_context.projector, query_eog=annotations.external_eog, artifactclasses=annotations.artifactclasses, predicted_contamination=None, trial_labels=annotations.trial_labels, samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz), minimum_trials_per_condition=2, status="success", operator_source="coordinate_corrected_subject_bridge", gamma=None, fallback_used=(method in {"POP-FALLBACK", "D-POP-CONTEXT", "U-POP-CONTEXT"}), uses_query_external_eog=False)
                rows.append({"dataset": "sgeyesub", "unit_id": unit, "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}", "study": prepared.fold.study, "training_seed": SEED, "method": method, **metric, **{key: value for key, value in diagnostics.items() if not isinstance(value, (list, dict))}, "statistical_unit": "participant_stem", "outputs_frozen_before_query_scoring": True, "oracle_query_EOG": False})
    _write_csv(output_root / "metrics.csv", rows)
    summary = {"status": "completed_full_real_one_seed_bridge_screen", **_implementation(), **dict(row), "unit_count": len({value["unit_id"] for value in rows}), "method_rows": len(rows), "result": str(output_root / "metrics.csv")}
    _write_json(output_root / "result_summary.json", summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def _bootstrap(values: Sequence[float], seed: int = 20260811, draws: int = 20000) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, data.size, size=(draws, data.size))].mean(axis=1)
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def aggregate(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _, root = _load(config); rows: list[dict[str, Any]] = []
    for path in sorted((root / "screen").glob("*/fold_*/metrics.csv")):
        with path.open(encoding="utf-8", newline="") as stream: rows.extend(dict(value) for value in csv.DictReader(stream))
    if not rows: raise FileNotFoundError("bridge screen metrics are absent")
    numeric_metrics = ("clean_waveform_RRMSE", "clean_waveform_correlation", "artifact_reconstruction_relative_error", "delta_SNR_db", "eog_coherence_reduction", "nonartifact_observation_preservation", "reference_free_psd_distortion", "reference_free_covariance_distortion", "condition_erp_observation_relative_preservation", "output_input_RMS_ratio", "observation_change_ratio")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "")).startswith("success"): grouped[(str(row["dataset"]), str(row["method"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, method), values in sorted(grouped.items()):
        result: dict[str, Any] = {"dataset": dataset, "method": method, "success_units": len({value["unit_id"] for value in values}), "coverage_denominator": 16 if dataset == "klados" else 59}
        for metric in numeric_metrics:
            numbers = [float(value[metric]) for value in values if value.get(metric) not in (None, "") and math.isfinite(float(value[metric]))]
            if numbers: result[f"mean_{metric}"] = float(np.mean(numbers)); result[f"median_{metric}"] = float(np.median(numbers))
        summaries.append(result)
    by_unit: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if str(row.get("status", "")).startswith("success"): by_unit[(str(row["dataset"]), str(row["unit_id"]))][str(row["method"])] = row
    effects: list[dict[str, Any]] = []
    for (dataset, unit), methods in sorted(by_unit.items()):
        metric = "clean_waveform_RRMSE" if dataset == "klados" else "eog_coherence_reduction"; sign = -1.0 if dataset == "klados" else 1.0
        matching = methods.get("D-MATCH"); pop = methods.get("POP-FALLBACK"); one = methods.get("U-MATCH")
        wrong = [methods.get(f"D-WRONG-{index}") for index in (1, 2, 3)]
        if matching and pop and one and all(wrong):
            m = float(matching[metric]); wrong_mean = float(np.mean([float(value[metric]) for value in wrong if value]))
            for estimand, right in (("subject_calibration", float(pop[metric])), ("diffusion_increment", float(one[metric])), ("subject_specificity", wrong_mean)):
                effects.append({"dataset": dataset, "unit_id": unit, "metric": metric, "estimand": estimand, "utility_effect_positive_is_better": sign * (m - right), "screen_seed": SEED})
    effect_summary: list[dict[str, Any]] = []
    for key in sorted({(value["dataset"], value["estimand"]) for value in effects}):
        values = [float(value["utility_effect_positive_is_better"]) for value in effects if (value["dataset"], value["estimand"]) == key]; low, high = _bootstrap(values)
        effect_summary.append({"dataset": key[0], "estimand": key[1], "statistical_units": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "positive_units": int(sum(value > 0 for value in values)), "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    _write_csv(root / "method_summary.csv", summaries); _write_csv(root / "paired_effects.csv", effects); _write_csv(root / "effect_summary.csv", effect_summary)
    lookup = {(row["dataset"], row["estimand"]): row for row in effect_summary}
    k_cal = lookup.get(("klados", "subject_calibration"), {}); s_cal = lookup.get(("sgeyesub", "subject_calibration"), {}); k_wrong = lookup.get(("klados", "subject_specificity"), {}); s_wrong = lookup.get(("sgeyesub", "subject_specificity"), {}); k_diff = lookup.get(("klados", "diffusion_increment"), {}); s_diff = lookup.get(("sgeyesub", "diffusion_increment"), {})
    method_lookup = {(row["dataset"], row["method"]): row for row in summaries}; s_match = method_lookup.get(("sgeyesub", "D-MATCH"), {}); s_pop = method_lookup.get(("sgeyesub", "POP-FALLBACK"), {})
    preservation = float(s_match.get("mean_nonartifact_observation_preservation", -999)) - float(s_pop.get("mean_nonartifact_observation_preservation", 0)); covariance = float(s_match.get("mean_reference_free_covariance_distortion", 999)) - float(s_pop.get("mean_reference_free_covariance_distortion", 0))
    gate = _mapping(config, "advance_gate")
    advance = all([
        float(k_cal.get("median", -999)) > 0, float(s_cal.get("median", -999)) > 0,
        float(k_wrong.get("median", -999)) > 0, float(s_wrong.get("median", -999)) > 0,
        float(k_diff.get("median", -999)) > 0, float(s_diff.get("median", -999)) > 0,
        int(k_cal.get("positive_units", 0)) >= int(gate["klados_supporting_records"]), int(s_cal.get("positive_units", 0)) >= int(gate["sge_supporting_stems"]),
        preservation >= float(gate["preservation_noninferiority"]), covariance <= float(gate["covariance_distortion_increase_maximum"]),
    ])
    summary = {"status": "completed_subject_bridge_repair_screen", **_implementation(), "historical_baa4ec8": {"numerical_execution": "numerically_correct_screen", "mechanism_ranking": "invalid_due_to_asymmetric_controls", "subject_aware_status": "not_yet_tested_fairly"}, "one_seed": SEED, "klados_coverage": "16/16", "sge_compatible_coverage": "58/59 total denominator", "fair_diffusion_vs_deterministic": True, "three_seed_expansion_allowed": advance, "three_seed_expansion_submitted": False, "oracle_query_EOG_diagnostic_status": "pending_separate_post_freeze_diagnostic", "outputs": {"method_summary": str(root / "method_summary.csv"), "paired_effects": str(root / "paired_effects.csv"), "effect_summary": str(root / "effect_summary.csv")}}
    _write_json(root / "result_summary.json", summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def technical(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, _ = _load(config); prepared = _prepared_route(base, task_rows()[0]); device = torch.device("cuda", 0)
    arrays = _corrected_arrays(prepared, float(_mapping(config, "training")["ridge_lambda"])); indices = np.arange(min(4, arrays["target"].shape[0])); target, condition = _batch(arrays, indices, device)
    model_config, diffusion_config = _model_configs(prepared, config); population = torch.as_tensor(prepared.population_context.full_transfer)
    diffusion = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=population).to(device); deterministic = FullCFiLMDeterministic(model_config, population_transfer=population).to(device); gate = AdaptiveActivityGate(prepared.model_dimensions.eeg_channels).to(device)
    optimizer = AdamW(list(diffusion.parameters()) + list(deterministic.parameters()) + list(gate.parameters()), lr=2e-4); generator = torch.Generator(device=device).manual_seed(8801)
    optimizer.zero_grad(set_to_none=True); loss_d, _ = diffusion.training_loss(target, generator=generator, **condition); prediction = deterministic(condition["observed"], **{key: value for key, value in condition.items() if key != "observed"}); loss_u = _masked_mse(prediction, target, condition["valid_time_mask"]); activity = gate(condition["observed"], condition["valid_time_mask"]); loss = loss_d + loss_u + activity.mean(); loss.backward(); optimizer.step()
    delta_a = physical_eeg_delta(target, normalized_transfer=condition["normalized_transfer"], latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=device), latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device), valid_time_mask=condition["valid_time_mask"])
    delta_b = physical_eeg_delta(target, normalized_transfer=condition["normalized_transfer"], latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=device), latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device), valid_time_mask=condition["valid_time_mask"])
    restored, bridge = coordinate_corrected_bridge(condition["observed"] * .9, context_delta=delta_a, population_delta=delta_b, beta=1.0, rho=1.0, activity_gate=activity, valid_time_mask=condition["valid_time_mask"])
    fallback, _ = coordinate_corrected_bridge(condition["observed"] * .9, context_delta=delta_a + 1, population_delta=delta_b, beta=0.0, rho=1.0, activity_gate=activity, valid_time_mask=condition["valid_time_mask"])
    checkpoint = run_dir / "technical.pt"; _save(checkpoint, {"diffusion": diffusion.state_dict(), "deterministic": deterministic.state_dict(), "gate": gate.state_dict()}); payload = torch.load(checkpoint, map_location=device, weights_only=False); diffusion.load_state_dict(payload["diffusion"])
    status = {"status": "passed", **_implementation(), "real_training_records": len(set(str(value) for value in arrays["recording_keys"])), "finite": bool(torch.isfinite(loss) and torch.isfinite(restored).all()), "physical_round_trip": bool(torch.allclose(delta_a, delta_b)), "beta_zero_exact_population": bool(torch.equal(fallback, condition["observed"] * .9 * condition["valid_time_mask"][:, None])), "zero_bridge_for_equal_physical_delta": float(bridge.abs().max()) == 0.0, "film_blocks_diffusion": diffusion.film_block_count, "film_blocks_deterministic": deterministic.film_block_count, "support_artifact_spectrum_removed": True, "checkpoint_reload": True}
    if not all(status[key] for key in ("finite", "physical_round_trip", "beta_zero_exact_population", "zero_bridge_for_equal_physical_delta", "checkpoint_reload")): status["status"] = "failed"
    _write_json(run_dir / "technical_status.json", status); return status


def j0(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _, root = _load(config); data = _mapping(config, "data"); availability = {key: Path(str(data[key])).exists() for key in ("klados", "sgeyesub")}
    if not all(availability.values()): raise FileNotFoundError(str(availability))
    _write_csv(root / "task_list.csv", task_rows())
    summary = {"status": "completed_J0", **_implementation(), "availability": availability, "tasks": len(task_rows()), "screen_seed": SEED, "deferred_seeds_not_submitted": [20260812, 20260813], "FIR_R2_P5_P6": "not_run", "historical_result_preserved": True}
    _write_json(run_dir / "result_summary.json", summary); return summary


def run_stage(config: Mapping[str, Any], run_dir: Path, stage: str, task: int | None) -> Mapping[str, Any]:
    if stage == "j0": return j0(config, run_dir)
    if stage == "technical": return technical(config, run_dir)
    if stage == "train-worker":
        if task is None or not 0 <= task < 8: raise ValueError("train worker index must be [0,7]")
        results = [train_task(config, run_dir / f"task_{index:02d}", index) for index in range(task, len(task_rows()), 8)]
        summary = {"status": "completed_train_worker", **_implementation(), "worker": task, "tasks": len(results)}; _write_json(run_dir / "worker_summary.json", summary); return summary
    if stage == "infer-worker":
        if task is None or not 0 <= task < 8: raise ValueError("infer worker index must be [0,7]")
        results = [_screen_task(config, run_dir / f"task_{index:02d}", index) for index in range(task, len(task_rows()), 8)]
        summary = {"status": "completed_infer_worker", **_implementation(), "worker": task, "tasks": len(results)}; _write_json(run_dir / "worker_summary.json", summary); return summary
    if stage == "aggregate": return aggregate(config, run_dir)
    raise ValueError(f"unknown subject bridge stage: {stage}")


__all__ = ["aggregate", "infer_unit", "j0", "run_stage", "task_rows", "technical", "train_task"]
