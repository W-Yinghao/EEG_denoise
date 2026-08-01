"""Single-GPU real-EEG integration validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.experiments.common import (
    _checkpoint_contract,
    _make_scaler,
    build_prior,
    configure_reproducibility,
    load_prior_data,
)
from eeg_cgdr.experiments.klados import calibration_batch, prepare_query
from eeg_cgdr.inference import (
    InformationMatchedOneStep,
    PopulationOnlyInference,
    attenuation_from_external_reference,
    matched_population_and_context_states,
    population_state_only,
)
from eeg_cgdr.operators import P0Config, fit_p0
from eeg_cgdr.training import resume_training_checkpoint, save_training_checkpoint


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    klados = config["klados"]
    return {
        "data_root": klados["data_root"],
        "files": {
            "contaminated": klados["contaminated"],
            "clean": klados["clean"],
            "heog": klados["heog"],
            "veog": klados["veog"],
        },
        "official_description": {"records": 54},
    }


def _frozen_source_split(config: dict[str, Any]) -> dict[str, float | int]:
    """Read the bounded Klados support/query contract used by integration."""
    path = Path(config["klados"]["split_manifest"])
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    support = [row for row in rows if row.get("status") == "held_out_calibration"]
    query = [row for row in rows if row.get("status") == "held_out_query"]
    if len(rows) != 2 or len(support) != 1 or len(query) != 1:
        raise ValueError("Klados integration requires one frozen support/query pair")
    support_row, query_row = support[0], query[0]
    if support_row["record"] != query_row["record"]:
        raise ValueError("Klados frozen support/query source records differ")
    record = support_row["record"]
    if not record.startswith("sim") or not record[3:].isdigit():
        raise ValueError("invalid frozen Klados source-record ID")
    source_rate = int(float(support_row["sampling_rate"]))
    if float(source_rate) != float(support_row["sampling_rate"]):
        raise ValueError("Klados frozen sampling rate is not integral")
    calibration_end = float(support_row["calibration_end"])
    query_start = float(query_row["query_start"])
    query_end = float(query_row["query_end"])
    if calibration_end + float(config["klados"]["guard_seconds"]) > query_start:
        raise ValueError("Klados integration support/query guard overlaps")
    return {
        "record_id": int(record[3:]),
        "source_rate": source_rate,
        "calibration_end": calibration_end,
        "query_start": query_start,
        "query_end": query_end,
    }


def _p0_config(config: dict[str, Any]) -> P0Config:
    raw = config["p0"]
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=32,
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=float(raw["minimum_bootstrap_success"]),
        maximum_bootstrap_median_distance=float(raw["maximum_bootstrap_median_distance"]),
        maximum_bootstrap_q90_distance=float(raw["maximum_bootstrap_q90_distance"]),
    )


def run_gpu_integration(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("GPU integration requires CUDA")
    population_base = float(config["observation"]["population_precision"])
    context_base = float(config["observation"]["context_precision"])
    if population_base != context_base:
        raise AssertionError("E0 and EC must use the same base precision")
    guidance_scale = float(config["observation"]["guidance_step"])
    seed = int(config["seed"])
    configure_reproducibility(seed)
    split = load_prior_data(config)
    prior = build_prior(config, device)
    optimizer = AdamW(prior.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    scaler = _make_scaler(True)
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed + 500)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 501)
    generators = {"loader": loader_generator, "training_noise": noise_generator}

    batch_size = 32
    update_losses: list[float] = []
    prior.train()
    order = torch.randperm(split.train.shape[0], generator=loader_generator)
    for update in range(20):
        start = (update * batch_size) % order.numel()
        index = order[start : start + batch_size]
        if index.numel() < batch_size:
            index = torch.cat([index, order[: batch_size - index.numel()]])
        clean = torch.from_numpy(split.train[index.numpy(), None, :]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = prior.training_loss(clean, generator=noise_generator)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite integration loss at update {update}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_losses.append(float(loss.detach()))

    checkpoint = run_dir / "integration_checkpoint.pt"
    contract = _checkpoint_contract(config)
    normalizer = {
        "mean": float(split.mean),
        "standard_deviation": float(split.standard_deviation),
        "sampling_rate": int(split.sampling_rate),
    }
    save_training_checkpoint(
        checkpoint,
        model=prior,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        step=20,
        config=contract,
        normalizer=normalizer,
        generators=generators,
        extra={"integration_updates": 20},
    )

    reloaded = build_prior(config, device)
    reloaded_optimizer = AdamW(reloaded.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    reloaded_scaler = _make_scaler(True)
    reloaded_noise_generator = torch.Generator(device=device)
    reloaded_noise_generator.manual_seed(0)
    reloaded_loader_generator = torch.Generator(device="cpu")
    reloaded_loader_generator.manual_seed(0)
    reloaded_generators = {
        "loader": reloaded_loader_generator,
        "training_noise": reloaded_noise_generator,
    }
    resumed = resume_training_checkpoint(
        checkpoint,
        model=reloaded,
        optimizer=reloaded_optimizer,
        scaler=reloaded_scaler,
        generators=reloaded_generators,
        expected_config=contract,
        map_location=device,
    )
    if resumed.step != 20 or resumed.epoch != 0:
        raise AssertionError("checkpoint cursor did not reload")
    next_start = 20 * batch_size
    next_index = order[next_start : next_start + batch_size]
    clean = torch.from_numpy(split.train[next_index.numpy(), None, :]).to(device)
    next_step_cpu_rng = torch.get_rng_state()
    next_step_cuda_rng = torch.cuda.get_rng_state_all()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        uninterrupted_loss = prior.training_loss(clean, generator=noise_generator)
    scaler.scale(uninterrupted_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # Dropout consumes the global RNG rather than the named diffusion-noise
    # generator.  Restore the checkpoint-time global streams before executing
    # the independently reloaded continuation.
    torch.set_rng_state(next_step_cpu_rng)
    torch.cuda.set_rng_state_all(next_step_cuda_rng)
    reloaded_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        resumed_loss = reloaded.training_loss(clean, generator=reloaded_noise_generator)
    reloaded_scaler.scale(resumed_loss).backward()
    reloaded_scaler.unscale_(reloaded_optimizer)
    torch.nn.utils.clip_grad_norm_(reloaded.parameters(), 1.0)
    reloaded_scaler.step(reloaded_optimizer)
    reloaded_scaler.update()
    if not bool(torch.isfinite(resumed_loss)):
        raise FloatingPointError("non-finite loss after checkpoint resume")
    if not torch.allclose(
        uninterrupted_loss.detach(), resumed_loss.detach(), atol=1.0e-7, rtol=1.0e-6
    ):
        raise AssertionError("resumed next-step loss differs from uninterrupted training")
    resume_parameter_max_difference = 0.0
    for name, reference_parameter in prior.named_parameters():
        resumed_parameter = dict(reloaded.named_parameters())[name]
        difference = float(
            torch.max(
                torch.abs(reference_parameter.detach() - resumed_parameter.detach())
            )
        )
        resume_parameter_max_difference = max(
            resume_parameter_max_difference, difference
        )
        if not torch.allclose(
            reference_parameter.detach(),
            resumed_parameter.detach(),
            atol=1.0e-7,
            rtol=1.0e-6,
        ):
            raise AssertionError(
                f"resumed next-step parameter differs from uninterrupted training: {name}"
            )

    for parameter in reloaded.parameters():
        parameter.requires_grad_(False)
    reloaded.eval()
    records = load_klados_records(_loader_config(config))
    frozen_split = _frozen_source_split(config)
    source_rate = int(frozen_split["source_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    p0_config = _p0_config(config)
    inference = PopulationOnlyInference(reloaded)
    one_step = InformationMatchedOneStep(reloaded)
    source_results: list[dict[str, Any]] = []
    eligible_p0_branches = 0
    held_out_record_id = int(frozen_split["record_id"])
    wrong_record_id = int(config["klados"]["wrong_source_record"])
    for record_id in (held_out_record_id, wrong_record_id):
        record = records[record_id - 1]
        five_second_support = calibration_batch(
            record,
            duration_seconds=5.0,
            source_rate=source_rate,
            target_rate=target_rate,
            source_label=f"sim{record_id}",
        )
        five_second_outcome = fit_p0(
            five_second_support,
            p0_config,
            movement_threshold=float(config["p0"]["movement_threshold"]),
        )
        calibration_seconds = (
            float(frozen_split["calibration_end"])
            if record_id == held_out_record_id
            else 10.0
        )
        support = calibration_batch(
            record,
            duration_seconds=calibration_seconds,
            source_rate=source_rate,
            target_rate=target_rate,
            source_label=f"sim{record_id}",
        )
        outcome = fit_p0(
            support,
            p0_config,
            movement_threshold=float(config["p0"]["movement_threshold"]),
        )
        if outcome.transfer is not None:
            eligible_p0_branches += 1
        if record_id == held_out_record_id:
            query = prepare_query(
                record,
                source_rate=source_rate,
                target_rate=target_rate,
                query_start_seconds=float(frozen_split["query_start"]),
                query_end_seconds=float(frozen_split["query_end"]),
                window_samples=512,
                attenuation_scale=float(config["observation"]["attenuation_scale"]),
            )
            y_numpy = query.contaminated[0]
            eog_numpy = query.eog[0]
            query_seconds = float(query.valid_samples[0] / target_rate)
        else:
            twelve_seconds = calibration_batch(
                record,
                duration_seconds=12.0,
                source_rate=source_rate,
                target_rate=target_rate,
                source_label=f"sim{record_id}",
            )
            y_numpy = twelve_seconds.eeg[:, -512:]
            eog_numpy = twelve_seconds.eog[:, -512:]
            query_seconds = 2.0
        y_numpy = (y_numpy - split.mean) / split.standard_deviation
        eog_numpy = (eog_numpy - eog_numpy.mean(axis=1, keepdims=True)) / np.maximum(
            eog_numpy.std(axis=1, keepdims=True), 1.0e-8
        )
        y = torch.as_tensor(y_numpy[None], device=device, dtype=torch.float32)
        eog = torch.as_tensor(eog_numpy[None], device=device, dtype=torch.float32)
        attenuation = attenuation_from_external_reference(
            eog,
            scale=float(config["observation"]["attenuation_scale"]),
            floor=float(config["observation"]["attenuation_floor"]),
        )
        population = population_state_only(
            y,
            attenuation=attenuation,
            base_precision=population_base,
            guidance_scale=guidance_scale,
        )
        initial_noise = inference.make_initial_noise(population, seed=seed + record_id)
        direct_pop = inference.sample(
            population, initial_noise=initial_noise, ddim_steps=4
        )
        factory_calls = 0

        def forbidden_factory():
            nonlocal factory_calls
            factory_calls += 1
            raise AssertionError("rho=0 constructed a calibration state")

        rho_zero = inference.sample_cgdr(
            population,
            rho=0.0,
            calibration_accepted=True,
            context_state_factory=forbidden_factory,
            initial_noise=initial_noise,
            ddim_steps=4,
        )
        if factory_calls != 0 or not torch.equal(direct_pop, rho_zero):
            raise AssertionError("rho=0 is not state-wise identical to direct POP")
        if outcome.transfer is None:
            p0_result = inference.sample_cgdr(
                population,
                rho=1.0,
                calibration_accepted=False,
                context_state_factory=None,
                initial_noise=initial_noise,
                ddim_steps=4,
            )
            baseline = direct_pop
        else:
            matched_population, context = matched_population_and_context_states(
                y,
                attenuation=attenuation,
                projector=outcome.transfer.projector,
                base_precision=context_base,
                guidance_scale=guidance_scale,
            )
            if (
                not torch.equal(matched_population.observation, population.observation)
                or not torch.equal(matched_population.precision, population.precision)
                or matched_population.scale != population.scale
            ):
                raise AssertionError("POP/P0 did not use the same population state")
            p0_result = inference.sample_cgdr(
                population,
                rho=1.0,
                calibration_accepted=True,
                context_state_factory=lambda context=context: context,
                initial_noise=initial_noise,
                ddim_steps=4,
            )
            baseline = one_step.restore(
                observation=y,
                channel_precision=context.precision * float(context.scale),
                seed=seed + record_id,
                timestep=100,
            )
        for name, tensor in {
            "POP": direct_pop,
            "rho_zero": rho_zero,
            "P0": p0_result,
            "one_step": baseline,
        }.items():
            if tensor.shape != y.shape or not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"invalid {name} integration output")
        source_results.append(
            {
                "source_record": f"sim{record_id}",
                "calibration_seconds": calibration_seconds,
                "query_seconds": query_seconds,
                "five_second_p0_status": five_second_outcome.status,
                "five_second_p0_reasons": list(five_second_outcome.reasons),
                "five_second_fallback": five_second_outcome.fallback,
                "p0_rank": outcome.transfer.rank if outcome.transfer else None,
                "p0_status": outcome.status,
                "p0_reasons": list(outcome.reasons),
                "p0_fallback_used": outcome.transfer is None,
                "rho_zero_bitwise_equal_pop": True,
                "rho_zero_context_factory_calls": factory_calls,
                "pop_p0_mean_absolute_difference": float(
                    torch.mean(torch.abs(direct_pop - p0_result)).cpu()
                ),
                "one_step_finite": True,
            }
        )

    if eligible_p0_branches < 1:
        raise AssertionError("GPU integration never exercised an eligible real P0 branch")

    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "training_updates_before_checkpoint": 20,
        "training_updates_after_resume": 1,
        "initial_loss": update_losses[0],
        "last_pre_checkpoint_loss": update_losses[-1],
        "post_resume_loss": float(resumed_loss.detach()),
        "checkpoint": str(checkpoint),
        "checkpoint_reload": True,
        "checkpoint_resume": True,
        "checkpoint_resume_next_step_within_tolerance": True,
        "checkpoint_resume_parameter_max_abs_difference": resume_parameter_max_difference,
        "independent_source_records": 2,
        "sources": source_results,
    }
    (run_dir / "integration_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
