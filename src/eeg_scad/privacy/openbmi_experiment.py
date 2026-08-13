"""V36P two-stage OpenBMI exact-fiber external replication."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from .experiment import _loader, encode, evaluate_representation, seed_all, sha256
from .fiber import FiberOneStep, FiberSANDiff, HeadFiber
from .fiber_channel import FiberStratifiedResampler, compose_strong_release, head_aware_attacks, multisample_diagnostics, strong_model_replacement
from .fiber_experiment import _cpu_state, distribution_fidelity, exact_preservation, train_exact, train_stage_a
from .fiber_external import FiberGaussian, fit_membership_attack, training_exposure
from .leace import LEACE
from .models import EEGNetRepresentation
from .openbmi import N_CHANNELS, N_SAMPLES, OPENBMI_ROOT, load_openbmi, outer_folds


def make_eegnet() -> EEGNetRepresentation:
    return EEGNetRepresentation(channels=N_CHANNELS, samples=N_SAMPLES, task_classes=2)


def _task_ba(model: EEGNetRepresentation, data, device: torch.device) -> float:
    _, logits = encode(model, data, device)
    return float(balanced_accuracy_score(data.task, logits.argmax(1)))


def train_eegnet_stage_a(train, validation, device: torch.device, seed: int, output: Path):
    seed_all(seed)
    model = make_eegnet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 1
    best_state = None
    curve = []
    for epoch in range(80):
        model.train()
        losses = []
        for x, y in _loader(train.eeg, train.task, batch_size=64, shuffle=True, seed=seed + epoch):
            logits = model(x.to(device))
            loss = nn.functional.cross_entropy(logits, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        score = _task_ba(model, validation, device)
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_balanced_accuracy": score})
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = _cpu_state(model)
    if best_state is None:
        raise RuntimeError("EEGNet selection produced no checkpoint")
    model.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "selected_epoch": best_epoch, "validation_balanced_accuracy": best_score, "selection_stage": "A"}, output)
    return model, best_epoch, curve


def train_eegnet_exact(train, device: torch.device, seed: int, epochs: int, output: Path):
    seed_all(seed)
    model = make_eegnet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        for x, y in _loader(train.eeg, train.task, batch_size=64, shuffle=True, seed=seed + epoch):
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    state = _cpu_state(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "epochs": epochs, "selection_stage": "B_refit", "participants": 45, "session": "ses_0"}, output)
    return model


def _release_model(model, kind: str, geometry: HeadFiber, z_head: dict[str, np.ndarray], h: dict[str, np.ndarray], device: torch.device, seed: int):
    result: dict[str, np.ndarray] = {}
    replacements: dict[str, np.ndarray] = {}
    for index, key in enumerate(z_head):
        replacement = strong_model_replacement(model, kind, h[key], geometry.fiber_dim, device, seed + index)
        replacements[key] = replacement
        result[key] = compose_strong_release(geometry, z_head[key], replacement)
    return result, replacements


def run_openbmi_fold(result_root: Path, fold: int, seed: int, device: torch.device, data_root: Path = OPENBMI_ROOT) -> dict[str, object]:
    split = outer_folds()[fold]
    run = result_root / "runtime" / f"fold_{fold}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)

    # Stage A: participant-disjoint selection.
    train_a = load_openbmi(data_root, split["train_subjects"], "ses_0")
    val_gallery = load_openbmi(data_root, split["validation_subjects"], "ses_0")
    val_query = load_openbmi(data_root, split["validation_subjects"], "ses_1")
    stage_a_eegnet, eegnet_epoch, eegnet_curve = train_eegnet_stage_a(train_a, val_query, device, seed, run / "stage_a_eegnet.pt")
    za_train, _ = encode(stage_a_eegnet, train_a, device)
    za_gallery, _ = encode(stage_a_eegnet, val_gallery, device)
    za_query, _ = encode(stage_a_eegnet, val_query, device)
    geometry_a = HeadFiber.from_linear(stage_a_eegnet.task_head)
    z_a = {"train": za_train, "gallery": za_gallery, "query": za_query}
    decomposition_a = {key: geometry_a.decompose(value) for key, value in z_a.items()}
    task_a = {"train": train_a.task, "gallery": val_gallery.task, "query": val_query.task}
    subject_a = {"train": train_a.subject, "gallery": val_gallery.subject, "query": val_query.subject}
    one_a = FiberOneStep(geometry_a.fiber_dim, 2).to(device)
    sand_a = FiberSANDiff(geometry_a.fiber_dim, 2).to(device)
    one_selection = train_stage_a("Fiber-OneStep", one_a, geometry_a, z_a, task_a, subject_a, stage_a_eegnet.task_head, device, seed + 1000, fold, run / "stage_a" / "Fiber-OneStep.pt")
    sand_selection = train_stage_a("Fiber-SANDiff", sand_a, geometry_a, z_a, task_a, subject_a, stage_a_eegnet.task_head, device, seed + 2000, fold, run / "stage_a" / "Fiber-SANDiff.pt")
    del train_a, val_gallery, val_query, stage_a_eegnet, za_train, za_gallery, za_query, z_a, decomposition_a, one_a, sand_a
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage B: refit all 45 non-test participants on Session 1 only.
    full_subjects = sorted(split["train_subjects"] + split["validation_subjects"])
    full = load_openbmi(data_root, full_subjects, "ses_0")
    nontraining = load_openbmi(data_root, full_subjects, "ses_1")
    gallery = load_openbmi(data_root, split["test_subjects"], "ses_0")
    query = load_openbmi(data_root, split["test_subjects"], "ses_1")
    eegnet_path = run / "eegnet_full_pool.pt"
    eegnet = train_eegnet_exact(full, device, seed, eegnet_epoch, eegnet_path)
    z = {}
    z["train"], _ = encode(eegnet, full, device)
    z["nontraining"], _ = encode(eegnet, nontraining, device)
    z["gallery"], _ = encode(eegnet, gallery, device)
    z["query"], _ = encode(eegnet, query, device)
    geometry = HeadFiber.from_linear(eegnet.task_head)
    decomposition = {key: geometry.decompose(value) for key, value in z.items()}
    z_head = {key: value[0] for key, value in decomposition.items()}
    u = {key: value[1] for key, value in decomposition.items()}
    h = {key: value[2] for key, value in decomposition.items()}
    task = {"train": full.task, "nontraining": nontraining.task, "gallery": gallery.task, "query": query.task}
    subject = {"train": full.subject, "nontraining": nontraining.subject, "gallery": gallery.subject, "query": query.subject}

    one_path = run / "stage_b" / "Fiber-OneStep.pt"
    sand_path = run / "stage_b" / "Fiber-SANDiff.pt"
    one = train_exact("Fiber-OneStep", geometry.fiber_dim, h["train"], u["train"], device, seed + 1000, int(one_selection["selected_epoch"]), one_path)
    sand = train_exact("Fiber-SANDiff", geometry.fiber_dim, h["train"], u["train"], device, seed + 2000, int(sand_selection["selected_epoch"]), sand_path)
    gaussian = FiberGaussian.fit(u["train"], h["train"])
    gaussian_path = run / "stage_b" / "Fiber-Gaussian.npz"
    gaussian.save(gaussian_path)
    resampler = FiberStratifiedResampler.fit(u["train"], h["train"])
    leace = LEACE.fit(z["train"], subject["train"])

    release: dict[str, dict[str, np.ndarray]] = {
        "RAW": {key: z[key] for key in ("train", "gallery", "query")},
        "HEAD_ONLY": {key: z_head[key] for key in ("train", "gallery", "query")},
        "LEACE": {key: leace.transform(z[key]) for key in ("train", "gallery", "query")},
    }
    replacements: dict[str, dict[str, np.ndarray]] = {}
    release["Fiber-OneStep"], replacements["Fiber-OneStep"] = _release_model(one, "Fiber-OneStep", geometry, z_head, h, device, seed + 5000)
    release["Fiber-SANDiff"], replacements["Fiber-SANDiff"] = _release_model(sand, "Fiber-SANDiff", geometry, z_head, h, device, seed + 6000)
    gaussian_coverage = []
    resample_coverage = []
    release["Fiber-Gaussian"] = {}
    release["Fiber-Stratified-Resample"] = {}
    replacements["Fiber-Gaussian"] = {}
    replacements["Fiber-Stratified-Resample"] = {}
    for index, key in enumerate(z_head):
        replacement_g, coverage_g = gaussian.sample(h[key], seed=seed + 7000 + index)
        replacement_r, coverage_r = resampler.sample(h[key], seed=seed + 8000 + index)
        replacements["Fiber-Gaussian"][key] = replacement_g
        replacements["Fiber-Stratified-Resample"][key] = replacement_r
        release["Fiber-Gaussian"][key] = compose_strong_release(geometry, z_head[key], replacement_g)
        release["Fiber-Stratified-Resample"][key] = compose_strong_release(geometry, z_head[key], replacement_r)
        gaussian_coverage.append({"fold": fold, "seed": seed, "set": key, "queries": len(coverage_g), "model_only": True})
        routes = {route: sum(item["fallback_route"] == route for item in coverage_r) for route in ("exact_stratum", "class_fallback", "global_fallback")}
        resample_coverage.append({"fold": fold, "seed": seed, "set": key, "queries": len(coverage_r), **routes, "deployment_requires_training_fiber_bank": True})

    # Utility and source-subject privacy.
    metrics = []
    participant_effects = []
    attacks = []
    attack_participants = []
    exact = []
    fidelity = []
    conditional = []
    for method, sets in release.items():
        row, participants = evaluate_representation(method, seed, sets["train"], task["train"], sets["gallery"], task["gallery"], subject["gallery"], sets["query"], task["query"], subject["query"], eegnet.task_head, device, fold, "strong" if method.startswith("Fiber-") else "na")
        metrics.append(row)
        participant_effects.extend(participants)
        gallery_u = geometry.decompose(sets["gallery"])[1]
        query_u = geometry.decompose(sets["query"])[1]
        attack_rows, participant_rows = head_aware_attacks(method, fold, seed, {"H": h["gallery"], "Z": sets["gallery"], "U": gallery_u}, {"H": h["query"], "Z": sets["query"], "U": query_u}, subject["gallery"], subject["query"])
        attacks.extend(attack_rows)
        attack_participants.extend(participant_rows)
        if method in ("HEAD_ONLY", "Fiber-OneStep", "Fiber-Gaussian", "Fiber-Stratified-Resample", "Fiber-SANDiff"):
            check = exact_preservation(geometry, z["query"], sets["query"], task["query"])
            check["H_recovery_max_abs_error"] = check["max_centered_logit_error"]
            exact.append({"fold": fold, "seed": seed, "method": method, **check})
        if method.startswith("Fiber-"):
            fidelity.append({"fold": fold, "seed": seed, "method": method, **distribution_fidelity(u["query"], query_u, h["query"], seed)})

    for method in ("Fiber-OneStep", "Fiber-Gaussian", "Fiber-Stratified-Resample", "Fiber-SANDiff"):
        for family in ("linear", "adaptive_mlp"):
            lookup = {(row["attacker"], row["feature"]): row for row in attacks if row["method"] == method}
            conditional.append({
                "fold": fold, "seed": seed, "method": method, "attacker": family,
                "CE_A_H": lookup[(family, "A_H")]["cross_entropy"],
                "CE_A_HU": lookup[(family, "A_HU")]["cross_entropy"],
                "conditional_fiber_leakage": lookup[(family, "A_H")]["cross_entropy"] - lookup[(family, "A_HU")]["cross_entropy"],
                "interpretation": "finite cross-session closure diagnostic, not CMI",
            })

    # Sixteen registered releases per query; no target-selected sample.
    one_many = np.repeat(replacements["Fiber-OneStep"]["query"][None], 16, axis=0)
    gaussian_many = gaussian.sample_many(h["query"], releases=16, seed=seed + 10000)
    resample_many = resampler.sample_many(h["query"], releases=16, seed=seed + 11000)
    sand_many = np.stack([strong_model_replacement(sand, "Fiber-SANDiff", h["query"], geometry.fiber_dim, device, seed + 12000 + index) for index in range(16)])
    many = {
        "Fiber-OneStep": one_many,
        "Fiber-Gaussian": gaussian_many,
        "Fiber-Stratified-Resample": resample_many,
        "Fiber-SANDiff": sand_many,
    }
    diversity = [{"fold": fold, "seed": seed, **multisample_diagnostics(method, values, u["train"])} for method, values in many.items()]

    membership_attack = fit_membership_attack(u["train"], u["nontraining"], seed)
    exposure = []
    exposure_participants = []
    for method in ("Fiber-Gaussian", "Fiber-Stratified-Resample", "Fiber-SANDiff"):
        row, participants = training_exposure(method, many[method], u["train"], subject["train"], u["gallery"], membership_attack, fold=fold, seed=seed, query_subject=subject["query"])
        exposure.append(row)
        exposure_participants.extend(participants)

    bindings = [
        {"fold": fold, "seed": seed, "model": "OpenBMI_EEGNet", "path": str(eegnet_path.resolve()), "sha256": sha256(eegnet_path), "training_subjects": ";".join(map(str, full_subjects))},
        {"fold": fold, "seed": seed, "model": "Fiber-OneStep", "path": str(one_path.resolve()), "sha256": sha256(one_path), "training_subjects": ";".join(map(str, full_subjects))},
        {"fold": fold, "seed": seed, "model": "Fiber-Gaussian", "path": str(gaussian_path.resolve()), "sha256": sha256(gaussian_path), "training_subjects": ";".join(map(str, full_subjects))},
        {"fold": fold, "seed": seed, "model": "Fiber-SANDiff", "path": str(sand_path.resolve()), "sha256": sha256(sand_path), "training_subjects": ";".join(map(str, full_subjects))},
    ]
    payload = {
        "fold": fold,
        "seed": seed,
        "split": split,
        "selection_summary": [
            {"model": "EEGNet", "selected_epoch": eegnet_epoch, "validation_balanced_accuracy": max(row["validation_balanced_accuracy"] for row in eegnet_curve)},
            {"model": "Fiber-OneStep", "selected_epoch": one_selection["selected_epoch"], "validation_balance": one_selection["validation_balance"], "checkpoint_rule": "full_output"},
            {"model": "Fiber-SANDiff", "selected_epoch": sand_selection["selected_epoch"], "validation_balance": sand_selection["validation_balance"], "checkpoint_rule": "full_10_step"},
        ],
        "selection_curves": {"EEGNet": eegnet_curve, "Fiber-OneStep": one_selection["curve"], "Fiber-SANDiff": sand_selection["curve"]},
        "checkpoint_binding": bindings,
        "fiber_geometry": {"fold": fold, "seed": seed, **geometry.diagnostics()},
        "metrics": metrics,
        "participant_effects": participant_effects,
        "head_aware_attacks": attacks,
        "attack_participant_effects": attack_participants,
        "conditional_fiber_leakage": conditional,
        "exact_preservation": exact,
        "distribution_fidelity": fidelity,
        "multisample_diversity": diversity,
        "training_exposure": exposure,
        "exposure_participant_effects": exposure_participants,
        "gaussian_coverage": gaussian_coverage,
        "resample_coverage": resample_coverage,
        "deployment_state": {
            "Fiber-Gaussian": ["frozen task head", "fiber geometry", "conditional Gaussian parameters", "normalization metadata"],
            "Fiber-Stratified-Resample": ["frozen task head", "fiber geometry", "training fiber bank", "training-derived strata"],
            "Fiber-SANDiff": ["frozen task head", "fiber geometry", "model weights", "normalization metadata"],
        },
        "code_path_audit": {
            "strong_replacement_receives_source_U": False,
            "gaussian_deployment_requires_training_bank": False,
            "resample_donor_pool": "outer_non_test_Session_1",
            "resample_reads_test_bank": False,
            "sandiff_deployment_requires_training_bank": False,
            "outer_test_used_for_selection": False,
            "target_selected_multisample": False,
        },
        "participant_coverage": {"test_participants": len(split["test_subjects"]), "test_trials_session_2": len(query.task)},
        "latency_benchmark_run": False,
        "waveform_sealed_reads": 0,
    }
    (run / "fold_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["make_eegnet", "run_openbmi_fold", "train_eegnet_exact", "train_eegnet_stage_a"]
