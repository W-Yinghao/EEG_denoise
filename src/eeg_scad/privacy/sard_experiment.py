"""V38P SARD-Bridge frozen-representation outer-fold experiment."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .experiment import encode, evaluate_representation, seed_all, sha256
from .fiber_experiment import _energy_distance, _mmd_rbf
from .fiber_external import fit_membership_attack, training_exposure
from .leace import LEACE
from .openbmi import OPENBMI_ROOT, OpenBMITrials, concatenate, load_openbmi, outer_folds
from .openbmi_experiment import make_eegnet
from .sard_bridge import (
    BridgeScaler, ContextNormalizer, DonorBank, FrozenSemantics, GaussianBridge,
    OneStepBridge, SARDBridge, SourceAdversary, probabilities, support_context,
)


V36_ROOT = Path("/home/infres/yinwang/denoiseNet_fiber_openbmi_v36p")
V36_BINDING = V36_ROOT / "results/fiber_openbmi_v36p/checkpoint_binding.csv"
METHODS = ("RAW", "LEACE", "OneStep-Bridge", "Gaussian-Bridge", "Stratified-Resample", "SARD-Bridge")
STRENGTHS = {"weak": .33, "medium": .66, "strong": 1.0}


def _subset(data: OpenBMITrials, mask: np.ndarray) -> OpenBMITrials:
    return OpenBMITrials(**{name: getattr(data, name)[mask] for name in OpenBMITrials.__dataclass_fields__})


def split_support(data: OpenBMITrials, per_class: int = 10) -> tuple[OpenBMITrials, OpenBMITrials]:
    """Chronological 10/class support and disjoint remaining Session-1 gallery."""
    chosen = np.zeros(len(data.task), dtype=bool)
    for owner in np.unique(data.subject):
        for task in (0, 1):
            candidates = np.flatnonzero((data.subject == owner) & (data.task == task))
            candidates = candidates[np.argsort(data.trial[candidates])]
            if len(candidates) < per_class:
                raise ValueError(f"participant {owner} has only {len(candidates)} trials for class {task}")
            chosen[candidates[:per_class]] = True
    return _subset(data, chosen), _subset(data, ~chosen)


def _load_frozen_eegnet(fold: int, seed: int, device: torch.device):
    rows = pd.read_csv(V36_BINDING)
    row = rows[(rows.fold == fold) & (rows.seed == seed) & (rows.model == "OpenBMI_EEGNet")]
    if len(row) != 1:
        raise ValueError(f"missing unique V36P EEGNet binding for fold={fold}, seed={seed}")
    item = row.iloc[0]; path = Path(item.path)
    if not path.is_file() or sha256(path) != item.sha256:
        raise ValueError(f"V36P checkpoint mismatch: {path}")
    model = make_eegnet().to(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, {"fold": fold, "seed": seed, "model": "V36P_EEGNet", "path": str(path), "sha256": str(item.sha256)}


def _contexts(
    support_z: np.ndarray, support_logits: np.ndarray, support_subject: np.ndarray,
    semantics_model: FrozenSemantics, training_subjects: list[int], normalizer: ContextNormalizer | None = None,
) -> tuple[dict[int, np.ndarray], ContextNormalizer]:
    owners, values = support_context(support_z, support_logits, support_subject, semantics_model)
    raw = {int(owner): value for owner, value in zip(owners, values)}
    if normalizer is None:
        normalizer = ContextNormalizer.fit(np.stack([raw[int(owner)] for owner in training_subjects]))
    return {owner: normalizer.transform(value) for owner, value in raw.items()}, normalizer


def _expand_context(mapping: dict[int, np.ndarray], subject: np.ndarray) -> np.ndarray:
    return np.stack([mapping[int(owner)] for owner in subject]).astype(np.float32)


def _head_logits(head: nn.Module, z: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(z)).float().to(device)).cpu().numpy()


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _train_bridge(
    kind: str, z: np.ndarray, logits: np.ndarray, subject: np.ndarray, context: np.ndarray,
    val_z: np.ndarray, val_logits: np.ndarray, val_subject: np.ndarray, val_context: np.ndarray,
    donor: DonorBank, head: nn.Module, device: torch.device, seed: int, output: Path,
    epochs: int = 30, batch_size: int = 256, lambda_task: float = .5, lambda_source: float = .1,
):
    seed_all(seed)
    first_delta, _, _ = donor.sample(z, logits, subject, seed)
    scaler = BridgeScaler.fit(z, logits, first_delta)
    condition = scaler.condition(z, logits, context)
    val_condition = scaler.condition(val_z, val_logits, val_context)
    model = (OneStepBridge(condition.shape[1], z.shape[1]) if kind == "one" else SARDBridge(condition.shape[1], z.shape[1])).to(device)
    owners = sorted(np.unique(subject)); owner_map = {owner: index for index, owner in enumerate(owners)}
    mapped = np.asarray([owner_map[int(owner)] for owner in subject], dtype=np.int64)
    adversary = SourceAdversary(z.shape[1], len(owners)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    adversary_optimizer = torch.optim.AdamW(adversary.parameters(), lr=5e-4, weight_decay=1e-4)
    best = float("inf"); best_state = None; best_epoch = 0; curve = []
    head.eval()
    for epoch in range(epochs):
        delta, _, routes = donor.sample(z, logits, subject, seed + epoch * 7919)
        target = scaler.normalize_delta(delta)
        dataset = TensorDataset(
            torch.from_numpy(condition), torch.from_numpy(target), torch.from_numpy(z),
            torch.from_numpy(logits), torch.from_numpy(mapped),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
        model.train(); losses = []
        for cond_b, target_b, source_b, source_logits_b, owner_b in loader:
            cond_b, target_b, source_b, source_logits_b, owner_b = [value.to(device) for value in (cond_b, target_b, source_b, source_logits_b, owner_b)]
            if kind == "one": predicted = model(cond_b)
            else:
                timestep = torch.randint(0, len(model.alpha_bar), (len(cond_b),), device=device)
                predicted = model(model.q_sample(target_b, timestep, torch.randn_like(target_b)), cond_b, timestep)
            release = source_b + scaler.restore_delta(predicted)
            adversary_optimizer.zero_grad(set_to_none=True)
            adversary_loss = nn.functional.cross_entropy(adversary(release.detach()), owner_b)
            adversary_loss.backward(); adversary_optimizer.step()
            for parameter in adversary.parameters(): parameter.requires_grad_(False)
            task_loss = nn.functional.mse_loss(head(release), source_logits_b)
            privacy_loss = -nn.functional.cross_entropy(adversary(release), owner_b)
            loss = nn.functional.mse_loss(predicted, target_b) + lambda_task * task_loss + lambda_source * privacy_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            for parameter in adversary.parameters(): parameter.requires_grad_(True)
            losses.append(float(loss.detach()))
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            model.eval(); generator = torch.Generator(device=device).manual_seed(seed + 100000 + epoch)
            with torch.no_grad():
                cond_tensor = torch.from_numpy(val_condition).to(device)
                if kind == "one": normalized = model(cond_tensor)
                else: normalized = model.sample(cond_tensor, torch.randn((len(val_z), z.shape[1]), device=device, generator=generator), steps=10)
                release = torch.from_numpy(val_z).to(device) + scaler.restore_delta(normalized)
                task_error = nn.functional.mse_loss(head(release), torch.from_numpy(val_logits).to(device)).item()
                movement = torch.mean((release - torch.from_numpy(val_z).to(device)) ** 2).item()
                score = task_error + .01 * abs(movement - float(np.mean(first_delta ** 2)))
            curve.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "full_output_validation_score": score, "task_error": task_error, "movement_mse": movement, "exact_donor_route_rate": float(np.mean(np.asarray(routes) == "exact_stratum"))})
            if score < best:
                best, best_state, best_epoch = score, _state(model), epoch + 1
    if best_state is None: raise RuntimeError("bridge checkpoint selection failed")
    model.load_state_dict(best_state); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "kind": kind, "seed": seed, "selected_epoch": best_epoch, "full_output_validation_score": best, "scaler": asdict(scaler), "curve": curve}, output)
    return model, scaler, curve, best_epoch


def _sample_model(model, kind: str, scaler: BridgeScaler, z: np.ndarray, logits: np.ndarray, context: np.ndarray, count: int, seed: int, device: torch.device) -> np.ndarray:
    condition = torch.from_numpy(scaler.condition(z, logits, context)).to(device)
    source = torch.from_numpy(z).to(device); releases = []
    model.eval()
    with torch.no_grad():
        for draw in range(count):
            if kind == "one": normalized = model(condition)
            else:
                generator = torch.Generator(device=device).manual_seed(seed + draw)
                normalized = model.sample(condition, torch.randn((len(z), z.shape[1]), device=device, generator=generator), steps=10)
            releases.append((source + scaler.restore_delta(normalized)).cpu().numpy())
    return np.stack(releases)


def _distribution(method: str, releases: np.ndarray, source_logits: np.ndarray, source_subject: np.ndarray, donor: DonorBank, seed: int, fold: int) -> dict[str, object]:
    target_delta = donor.sample_many(np.zeros_like(releases[0]), source_logits, source_subject, releases.shape[0], seed)
    # sample_many above returns donor z because source is zero; target distribution is valid donor representations.
    generated = releases.reshape(-1, releases.shape[-1]); target = target_delta.reshape(-1, target_delta.shape[-1])
    rng = np.random.default_rng(seed); cap = min(768, len(generated), len(target))
    gi = rng.choice(len(generated), cap, replace=False); ti = rng.choice(len(target), cap, replace=False)
    g, t = generated[gi].astype(np.float64), target[ti].astype(np.float64)
    scale = np.std(t, axis=0); scale = np.where(scale > 1e-6, scale, 1.0); g /= scale; t /= scale
    covariance = np.linalg.norm(np.cov(g, rowvar=False) - np.cov(t, rowvar=False), "fro") / max(np.linalg.norm(np.cov(t, rowvar=False), "fro"), 1e-8)
    generated_var = np.var(generated, axis=0, ddof=1).sum(); target_var = np.var(target, axis=0, ddof=1).sum()
    within = float(np.var(releases, axis=0, ddof=1).sum(1).mean()) if releases.shape[0] > 1 else 0.0
    between = float(np.var(releases.mean(0), axis=0, ddof=1).sum())
    duplicate = float(np.mean(np.linalg.norm(releases[1:] - releases[:-1], axis=2) < 1e-7)) if releases.shape[0] > 1 else 1.0
    return {"fold": fold, "method": method, "conditional_energy_distance": _energy_distance(g, t), "conditional_mmd_rbf": _mmd_rbf(g, t), "conditional_covariance_discrepancy": float(covariance), "variance_retained": float(generated_var / max(target_var, 1e-8)), "within_query_diversity": within, "between_query_variation": between, "duplicate_rate": duplicate, "releases_per_query": int(releases.shape[0])}


def _ensemble_and_augmentation(method: str, train_release: np.ndarray, train_task: np.ndarray, query_release: np.ndarray, query_task: np.ndarray, query_subject: np.ndarray, head: nn.Module, device: torch.device, fold: int, seed: int):
    probabilities_k = np.stack([probabilities(_head_logits(head, draw, device)) for draw in query_release])
    ensemble_prediction = probabilities_k.mean(0).argmax(1)
    train_flat = train_release.reshape(-1, train_release.shape[-1]); task_flat = np.tile(train_task, train_release.shape[0])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)).fit(train_flat, task_flat)
    augmented_probability = np.stack([probe.predict_proba(draw) for draw in query_release]).mean(0)
    row = {"fold": fold, "seed": seed, "method": method, "release_count": int(len(query_release)), "ensemble_frozen_head_balanced_accuracy": float(balanced_accuracy_score(query_task, ensemble_prediction)), "augmentation_retrained_head_balanced_accuracy": float(balanced_accuracy_score(query_task, augmented_probability.argmax(1)))}
    participants = []
    for owner in np.unique(query_subject):
        mask = query_subject == owner
        participants.append({"fold": fold, "seed": seed, "method": method, "participant": int(owner + 1), "ensemble_frozen_head_balanced_accuracy": float(balanced_accuracy_score(query_task[mask], ensemble_prediction[mask])), "augmentation_retrained_head_balanced_accuracy": float(balanced_accuracy_score(query_task[mask], augmented_probability[mask].argmax(1)))})
    return row, participants


def run_fold(result_root: Path, fold: int, seed: int, device: torch.device, data_root: Path = OPENBMI_ROOT) -> dict[str, object]:
    seed_all(seed + fold * 10000); split = outer_folds()[fold]
    runtime = result_root / "runtime" / f"fold_{fold}_seed_{seed}"; runtime.mkdir(parents=True, exist_ok=True)
    eegnet, eegnet_binding = _load_frozen_eegnet(fold, seed, device)
    groups = {"train": split["train_subjects"], "validation": split["validation_subjects"], "test": split["test_subjects"]}
    data = {}; support = {}; gallery = {}; query = {}
    for group, owners in groups.items():
        data[group] = load_openbmi(data_root, owners, "ses_0")
        support[group], gallery[group] = split_support(data[group])
        query[group] = load_openbmi(data_root, owners, "ses_1")
    encoded = {}
    for group in groups:
        encoded[(group, "support")] = encode(eegnet, support[group], device)
        encoded[(group, "gallery")] = encode(eegnet, gallery[group], device)
        encoded[(group, "query")] = encode(eegnet, query[group], device)
    train_z = np.concatenate([encoded[("train", "gallery")][0], encoded[("train", "query")][0]])
    train_logits = np.concatenate([encoded[("train", "gallery")][1], encoded[("train", "query")][1]])
    train_subject = np.concatenate([gallery["train"].subject, query["train"].subject])
    train_task = np.concatenate([gallery["train"].task, query["train"].task])
    val_z = np.concatenate([encoded[("validation", "gallery")][0], encoded[("validation", "query")][0]])
    val_logits = np.concatenate([encoded[("validation", "gallery")][1], encoded[("validation", "query")][1]])
    val_subject = np.concatenate([gallery["validation"].subject, query["validation"].subject])
    semantics_model = FrozenSemantics.fit(train_z, train_logits)
    context_map, context_normalizer = _contexts(encoded[("train", "support")][0], encoded[("train", "support")][1], support["train"].subject, semantics_model, split["train_subjects"])
    for group in ("validation", "test"):
        group_map, _ = _contexts(encoded[(group, "support")][0], encoded[(group, "support")][1], support[group].subject, semantics_model, split["train_subjects"], context_normalizer)
        context_map.update(group_map)
    train_context = _expand_context(context_map, train_subject); val_context = _expand_context(context_map, val_subject)
    donor = DonorBank.fit(train_z, train_logits, train_subject, semantics_model)
    one_path = runtime / "one_step_bridge.pt"; sard_path = runtime / "sard_bridge.pt"
    one, one_scaler, one_curve, one_epoch = _train_bridge("one", train_z, train_logits, train_subject, train_context, val_z, val_logits, val_subject, val_context, donor, eegnet.task_head, device, seed + fold * 1000 + 101, one_path)
    sard, sard_scaler, sard_curve, sard_epoch = _train_bridge("sard", train_z, train_logits, train_subject, train_context, val_z, val_logits, val_subject, val_context, donor, eegnet.task_head, device, seed + fold * 1000 + 202, sard_path)
    initial_delta, _, donor_routes = donor.sample(train_z, train_logits, train_subject, seed + 303)
    gaussian_scaler = BridgeScaler.fit(train_z, train_logits, initial_delta)
    gaussian_condition = gaussian_scaler.condition(train_z, train_logits, train_context)
    gaussian_targets = []
    for draw in range(4): gaussian_targets.append(gaussian_scaler.normalize_delta(donor.sample(train_z, train_logits, train_subject, seed + 400 + draw)[0]))
    gaussian = GaussianBridge.fit(np.tile(gaussian_condition, (4, 1)), np.concatenate(gaussian_targets), np.tile(train_logits, (4, 1)), semantics_model)
    leace = LEACE.fit(train_z, train_subject)
    releases: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
    set_specs = {
        "train": (train_z, train_logits, train_subject, train_task),
        "gallery": (encoded[("test", "gallery")][0], encoded[("test", "gallery")][1], gallery["test"].subject, gallery["test"].task),
        "query": (encoded[("test", "query")][0], encoded[("test", "query")][1], query["test"].subject, query["test"].task),
        "val_gallery": (encoded[("validation", "gallery")][0], encoded[("validation", "gallery")][1], gallery["validation"].subject, gallery["validation"].task),
        "val_query": (encoded[("validation", "query")][0], encoded[("validation", "query")][1], query["validation"].subject, query["validation"].task),
    }
    for index, (name, (z, logits, owner, _)) in enumerate(set_specs.items()):
        context = _expand_context(context_map, owner); count = 8
        releases["RAW"][name] = np.repeat(z[None], count, axis=0)
        releases["LEACE"][name] = np.repeat(leace.transform(z)[None], count, axis=0)
        releases["OneStep-Bridge"][name] = _sample_model(one, "one", one_scaler, z, logits, context, count, seed + 10000 + index * 100, device)
        condition = gaussian_scaler.condition(z, logits, context)
        gaussian_delta = gaussian.sample_many(condition, logits, count, seed + 20000 + index * 100)
        releases["Gaussian-Bridge"][name] = z[None] + np.stack([gaussian_scaler.restore_delta(torch.from_numpy(value)).numpy() for value in gaussian_delta])
        releases["Stratified-Resample"][name] = z[None] + donor.sample_many(z, logits, owner, count, seed + 30000 + index * 100)
        releases["SARD-Bridge"][name] = _sample_model(sard, "sard", sard_scaler, z, logits, context, count, seed + 40000 + index * 100, device)
    # Three-point participant-disjoint validation sweep; alpha scales transport, never uses outer test.
    validation = []
    for method in ("OneStep-Bridge", "Gaussian-Bridge", "Stratified-Resample", "SARD-Bridge"):
        raw_train = releases["RAW"]["train"][0]
        raw_gallery = releases["RAW"]["val_gallery"][0]; raw_query = releases["RAW"]["val_query"][0]
        for strength, alpha in STRENGTHS.items():
            transformed_train = raw_train + alpha * (releases[method]["train"][0] - raw_train)
            transformed_gallery = raw_gallery + alpha * (releases[method]["val_gallery"][0] - raw_gallery)
            transformed_query = raw_query + alpha * (releases[method]["val_query"][0] - raw_query)
            row, _ = evaluate_representation(method, seed, transformed_train, train_task, transformed_gallery, gallery["validation"].task, gallery["validation"].subject, transformed_query, query["validation"].task, query["validation"].subject, eegnet.task_head, device, fold, strength)
            row["selection_balance"] = row["fixed_head_balanced_accuracy"] + row["retrained_head_balanced_accuracy"] - .25 * row["adaptive_subject_attack_balanced_accuracy"] - .1 * abs(row["cross_session_same_different_auroc"] - .5)
            validation.append(row)
    selected = {method: max((row for row in validation if row["method"] == method), key=lambda row: row["selection_balance"])["strength"] for method in ("OneStep-Bridge", "Gaussian-Bridge", "Stratified-Resample", "SARD-Bridge")}
    selected.update({"RAW": "strong", "LEACE": "strong"})
    metrics = []; participants = []; fidelity = []; diversity = []; exposure = []; exposure_participants = []; ensemble = []; ensemble_participants = []
    membership = fit_membership_attack(train_z, val_z, seed)
    for method in METHODS:
        alpha = STRENGTHS.get(selected[method], 1.0)
        selected_release = {}
        for name in ("train", "gallery", "query"):
            raw = releases["RAW"][name]; selected_release[name] = raw + alpha * (releases[method][name] - raw)
        row, part = evaluate_representation(method, seed, selected_release["train"][0], train_task, selected_release["gallery"][0], gallery["test"].task, gallery["test"].subject, selected_release["query"][0], query["test"].task, query["test"].subject, eegnet.task_head, device, fold, selected[method])
        metrics.append(row); participants.extend(part)
        if method not in ("RAW", "LEACE"):
            dist = _distribution(method, selected_release["query"], encoded[("test", "query")][1], query["test"].subject, donor, seed + 50000, fold); fidelity.append(dist); diversity.append({key: value for key, value in dist.items() if key in ("fold", "method", "within_query_diversity", "between_query_variation", "duplicate_rate", "releases_per_query")})
            ens, ens_part = _ensemble_and_augmentation(method, selected_release["train"], train_task, selected_release["query"], query["test"].task, query["test"].subject, eegnet.task_head, device, fold, seed); ensemble.append(ens); ensemble_participants.extend(ens_part)
        if method in ("Gaussian-Bridge", "Stratified-Resample", "SARD-Bridge"):
            exp, exp_part = training_exposure(method, selected_release["query"], train_z, train_subject, val_z, membership, fold=fold, seed=seed, query_subject=query["test"].subject); exposure.append(exp); exposure_participants.extend(exp_part)
    checkpoint = [eegnet_binding, {"fold": fold, "seed": seed, "model": "OneStep-Bridge", "path": str(one_path), "sha256": sha256(one_path), "selected_epoch": one_epoch}, {"fold": fold, "seed": seed, "model": "SARD-Bridge", "path": str(sard_path), "sha256": sha256(sard_path), "selected_epoch": sard_epoch}]
    payload = {"fold": fold, "seed": seed, "split": split, "checkpoint_binding": checkpoint, "support": {"budget": 20, "per_class": 10, "gallery_per_participant": 80, "query_per_participant": 100}, "donor": {"bank_participants": split["train_subjects"], "bank_rows": len(train_z), "exact_stratum_rate": float(np.mean(np.asarray(donor_routes) == "exact_stratum")), "outer_test_rows": 0}, "validation": validation, "selected_strength": selected, "method_summary": metrics, "participant_effects": participants, "distribution_fidelity": fidelity, "multisample_diversity": diversity, "training_exposure": exposure, "exposure_participant_effects": exposure_participants, "ensemble_utility": ensemble, "ensemble_participant_effects": ensemble_participants, "training_curves": {"OneStep-Bridge": one_curve, "SARD-Bridge": sard_curve}, "repair_used": False, "sealed_reads": 0, "true_test_label_used_for_inference_or_donor_selection": False}
    (runtime / "fold_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["METHODS", "split_support", "run_fold"]
