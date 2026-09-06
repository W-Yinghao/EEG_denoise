"""SHU multi-session class-conditional task-phenotype development experiment."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eeg_cgdr.data.shu_mi import ShuMiStore, ShuTrialKey


RESULT = Path("results/cgdr/shu_task_phenotype_diffusion")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/shu_task_phenotype")
PHYSIO_ROOT = Path(
    "/home/infres/yinwang/denoiseNet_physiomotion_hybrid_masked/"
    "results/cgdr/physiomotion_hybrid_masked"
)
SHU_CHANNELS = (
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2",
)
FAMILIES = ("head_motion", "chewing", "tongue", "swallowing", "facial_emg")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def folds() -> dict[int, int]:
    """Five deterministic participant folds, fixed without signal access."""
    return {participant: (participant - 1) // 5 for participant in range(1, 26)}


def _normalize_name(name: str) -> str:
    value = name.strip().upper().replace(" ", "")
    return {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(value, value)


def _physio_channel_to_shu_indices(name: str) -> tuple[int, ...]:
    endpoints = [_normalize_name(piece) for piece in name.split("-")]
    return tuple(sorted({SHU_CHANNELS.index(v) for v in endpoints if v in SHU_CHANNELS}))


def freeze() -> dict[str, Any]:
    inventory = json.loads((RESULT / "j0" / "datalake_inventory.json").read_text())
    if not inventory["all_25x5_present"] or inventory["participants"] != 25:
        raise RuntimeError("SHU 25x5 coverage failed")
    if inventory["session_04_05_payloads_opened"]:
        raise RuntimeError("sealed SHU session payload was opened")

    split_rows = [
        {
            "participant": participant, "outer_fold": fold,
            "day_1_role": "support", "day_2_role": "development_query",
            "day_3_role": "development_query", "day_4_role": "session_sealed",
            "day_5_role": "session_sealed",
        }
        for participant, fold in folds().items()
    ]
    _csv(RESULT / "frozen" / "fold_session_manifest.csv", split_rows)

    source = PHYSIO_ROOT / "frozen" / "development_mask_source_audit.csv"
    audit_rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    source_participants = sorted({int(row["participant"]) for row in audit_rows})
    if len(source_participants) != 20:
        raise RuntimeError("mask source is not frozen PhysioMotion development-20")
    # Read only the already-opened development mask arrays, never waveforms.
    meta = json.loads((RESULT / "j0" / "source_metadata.json").read_text())
    physio_channels = list(meta["physiomotion"]["channels"])
    signatures: dict[str, set[tuple[tuple[int, ...], int, int]]] = defaultdict(set)
    fair_root = Path(
        "/home/infres/yinwang/denoiseNet_physiomotion_retrieval_fairness/"
        "results/cgdr/physiomotion_retrieval_fairness/fair_materialized"
    )
    split = list(csv.DictReader(
        (PHYSIO_ROOT.parent / "physiomotion_subject_restoration" / "metadata" / "frozen_participant_split.csv").open()
    ))
    development = sorted(int(r["participant"]) for r in split if r["role"] == "development")
    if development != source_participants:
        raise RuntimeError("PhysioMotion mask owner mismatch")
    for participant in development:
        path = fair_root / f"masks_{participant:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            for family, mask in zip(data["families"].astype(str), data["masks"].astype(bool)):
                if family not in FAMILIES or not mask.any():
                    continue
                affected_physio = np.flatnonzero(mask.any(axis=1))
                affected_shu: set[int] = set()
                for index in affected_physio:
                    affected_shu.update(_physio_channel_to_shu_indices(physio_channels[int(index)]))
                time = np.flatnonzero(mask.any(axis=0))
                if not affected_shu or not len(time):
                    continue
                duration = int(time[-1] - time[0] + 1)
                signatures[family].add((tuple(sorted(affected_shu)), int(time[0]), duration))

    mask_rows: list[dict[str, Any]] = []
    mask_arrays: list[np.ndarray] = []
    mask_families: list[str] = []
    for family_index, family in enumerate(FAMILIES):
        items = sorted(signatures[family], key=lambda x: (x[2], x[0], x[1]))
        # Freeze a compact, geometry-diverse library before any SHU result access.
        chosen = items if len(items) <= 8 else [items[round(i)] for i in np.linspace(0, len(items) - 1, 8)]
        for item_index, (channels, original_start, duration) in enumerate(chosen):
            mask = np.zeros((32, 1000), dtype=bool)
            start = 125 + ((family_index * 173 + item_index * 97) % max(1, 750 - duration))
            stop = min(1000, start + duration)
            mask[np.asarray(channels), start:stop] = True
            mask_arrays.append(mask); mask_families.append(family)
            mask_rows.append({
                "mask_id": len(mask_rows), "family": family,
                "source": "PhysioMotion development-20 annotation geometry",
                "source_start": original_start, "duration_samples_at_250hz": stop - start,
                "shu_start": start, "shu_stop": stop, "shu_channel_count": len(channels),
                "shu_channels": ";".join(SHU_CHANNELS[i] for i in channels),
            })
    # Fixed electrode-dropout geometries are part of the frozen library.
    for label, channels in (("single_electrode", (12,)), ("adjacent_double", (12, 13))):
        mask = np.zeros((32, 1000), dtype=bool); mask[list(channels), 375:625] = True
        mask_arrays.append(mask); mask_families.append(label)
        mask_rows.append({
            "mask_id": len(mask_rows), "family": label, "source": "preregistered electrode dropout",
            "source_start": "NA", "duration_samples_at_250hz": 250, "shu_start": 375,
            "shu_stop": 625, "shu_channel_count": len(channels),
            "shu_channels": ";".join(SHU_CHANNELS[i] for i in channels),
        })
    if not all(any(r["family"] == family for r in mask_rows) for family in FAMILIES):
        raise RuntimeError("one or more primary mask families failed channel mapping")
    DERIVED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DERIVED / "frozen_masks.npz", masks=np.asarray(mask_arrays, np.uint8),
        families=np.asarray(mask_families),
    )
    _csv(RESULT / "frozen" / "mask_manifest.csv", mask_rows)
    protocol = {
        "status": "FROZEN_BEFORE_DAY_2_3_SCIENTIFIC_EVALUATION",
        "data_asset": str(ShuMiStore().path), "asset_kind": "trial-level 256 Hz datalake derivative",
        "protocol_rate_hz": 250, "resampling": "scipy.signal.resample_poly 125/128",
        "native_edf_mat_directly_present": False,
        "provenance_limitation": "The datalake preserves complete trial/session keys and labels but not the source EDF/MAT containers.",
        "participants": 25, "sessions": 5, "development_sessions": [1, 2, 3],
        "session_sealed": [4, 5], "session_04_05_payloads_opened": False,
        "outer_folds": 5, "fold_size": 5, "channels": list(SHU_CHANNELS),
        "classes": {"0": "left/right MI source class 0", "1": "left/right MI source class 1"},
        "first_reference_samples": 125, "band_hz": [8, 30], "time_bin_samples": 125,
        "probe": {"rank": 16, "ridge": 0.05, "spatial_temporal_blend": 0.5},
        "mask_source": "PhysioMotion frozen development-20 geometry only; no waveform and no sealed-10 access",
    }
    _json(RESULT / "frozen" / "protocol.json", protocol)
    return {"status": "J0_PROTOCOL_FROZEN", "mask_count": len(mask_rows), "sealed_opened": False}


def materialize(subject: int) -> dict[str, Any]:
    if subject not in range(1, 26):
        raise ValueError(subject)
    store = ShuMiStore()
    keys = [k for k in store.inventory() if k.subject == subject and k.session <= 3]
    rows = [store.load(key) for key in keys]
    eeg = np.asarray([row.eeg_uv for row in rows], np.float32)
    labels = np.asarray([row.label for row in rows], np.int8)
    sessions = np.asarray([row.key.session for row in rows], np.int8)
    trials = np.asarray([row.key.trial for row in rows], np.int16)
    path = DERIVED / "development" / f"participant_{subject:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, eeg=eeg, labels=labels, sessions=sessions, trials=trials)
    summary = {
        "subject": subject, "trials": len(rows), "session_counts": {
            str(session): int(np.sum(sessions == session)) for session in (1, 2, 3)
        }, "shape": list(eeg.shape), "finite": bool(np.isfinite(eeg).all()),
        "sessions_04_05_opened": False,
    }
    _json(RESULT / "materialize" / f"participant_{subject:02d}.json", summary)
    return summary


def _load_subject(subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(DERIVED / "development" / f"participant_{subject:02d}.npz") as data:
        return np.asarray(data["eeg"], np.float32), np.asarray(data["labels"], int), np.asarray(data["sessions"], int)


def _bandpass(eeg: np.ndarray) -> np.ndarray:
    sos = butter(4, (8, 30), btype="bandpass", fs=250, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1).astype(np.float32)


def _covariance(trials: np.ndarray, rank: int = 16, ridge: float = .05) -> np.ndarray:
    flat = np.transpose(trials, (1, 0, 2)).reshape(32, -1)
    flat -= flat.mean(axis=1, keepdims=True)
    covariance = flat @ flat.T / max(flat.shape[1] - 1, 1)
    scale = float(np.trace(covariance) / 32)
    covariance = (1 - ridge) * covariance + ridge * scale * np.eye(32)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    keep = order[:rank]
    floor = max(float(values[keep[-1]]) * .01, scale * 1e-6)
    return ((vectors[:, keep] * np.maximum(values[keep], floor)) @ vectors[:, keep].T + floor * np.eye(32)).astype(np.float64)


def _phenotype(trials: np.ndarray) -> dict[str, np.ndarray]:
    filtered = _bandpass(trials)
    power = np.mean(filtered.reshape(len(filtered), 32, 8, 125) ** 2, axis=-1) + 1e-8
    log_power = np.log(power)
    reference = log_power[:, :, 0:1]
    erd = log_power - reference
    return {
        "log_power": log_power.mean(axis=0).astype(np.float32),
        "erd": erd.mean(axis=0).astype(np.float32),
        "topography": log_power.mean(axis=(0, 2)).astype(np.float32),
        "covariance": _covariance(filtered),
    }


def _population_phenotype(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.mean([value[key] for value in values], axis=0) for key in values[0]}


def _temporal_interpolate(clean_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = clean_masked.copy(); grid = np.arange(clean_masked.shape[-1])
    for channel in np.flatnonzero(mask.any(axis=1)):
        observed = ~mask[channel]
        if observed.sum() >= 2:
            output[channel, mask[channel]] = np.interp(grid[mask[channel]], grid[observed], clean_masked[channel, observed])
    return output


def _reconstruct(
    clean: np.ndarray, mask: np.ndarray, covariance: np.ndarray, ridge: float = .05,
    *, operator_cache: dict[Any, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]] | None = None,
    cache_key: Any = None,
) -> np.ndarray:
    observed_signal = clean.copy(); observed_signal[mask] = 0
    temporal = _temporal_interpolate(observed_signal, mask)
    output = observed_signal.copy()
    # Group identical channel masks, allowing the annotation geometry to vary over time.
    operators = operator_cache.get(cache_key) if operator_cache is not None else None
    if operators is None:
        patterns: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for time in np.flatnonzero(mask.any(axis=0)):
            patterns[tuple(np.flatnonzero(mask[:, time]))].append(int(time))
        operators = []
        for missing_tuple, times in patterns.items():
            missing = np.asarray(missing_tuple, int)
            observed = np.asarray([i for i in range(32) if i not in set(missing_tuple)], int)
            times_array = np.asarray(times, int)
            weight = None
            if len(observed):
                coo = covariance[np.ix_(observed, observed)]
                scale = float(np.trace(coo) / max(len(observed), 1))
                weight = covariance[np.ix_(missing, observed)] @ np.linalg.pinv(coo + ridge * scale * np.eye(len(observed)))
            operators.append((missing, observed, times_array, weight))
        if operator_cache is not None: operator_cache[cache_key] = operators
    for missing, observed, times_array, weight in operators:
        if len(observed) and weight is not None:
            spatial = weight @ observed_signal[np.ix_(observed, times_array)]
            output[np.ix_(missing, times_array)] = .5 * spatial + .5 * temporal[np.ix_(missing, times_array)]
        else:
            output[np.ix_(missing, times_array)] = temporal[np.ix_(missing, times_array)]
    return output.astype(np.float32)


def _bin_log_power(value: np.ndarray) -> np.ndarray:
    bins = value.reshape(32, 8, 125)
    spectrum = np.fft.rfft(bins, axis=-1)
    frequencies = np.fft.rfftfreq(125, 1 / 250)
    selected = (frequencies >= 8) & (frequencies <= 30)
    return np.log(np.mean(np.abs(spectrum[..., selected]) ** 2, axis=-1) + 1e-8)


def _masked_metrics(clean: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    truth, estimate = clean[mask], restored[mask]
    denominator = float(np.sqrt(np.mean(truth ** 2)) + 1e-8)
    rrmse = float(np.sqrt(np.mean((truth - estimate) ** 2)) / denominator)
    corr = float(np.corrcoef(truth, estimate)[0, 1]) if len(truth) > 2 and np.std(estimate) > 0 else 0.0
    lp_clean, lp_out = _bin_log_power(clean), _bin_log_power(restored)
    erd_clean, erd_out = lp_clean - lp_clean[:, :1], lp_out - lp_out[:, :1]
    return {
        "rrmse": rrmse, "correlation": corr,
        "erd_distortion": float(np.mean(np.abs(erd_clean - erd_out))),
        "topography_distortion": float(np.mean(np.abs(lp_clean.mean(axis=1) - lp_out.mean(axis=1)))),
    }


def _csp_filters(class0: np.ndarray, class1: np.ndarray, count: int = 3) -> np.ndarray:
    c0, c1 = _covariance(_bandpass(class0), rank=32), _covariance(_bandpass(class1), rank=32)
    values, vectors = eigh(c1, c0 + c1)
    order = np.argsort(values)
    selected = np.r_[order[:count], order[-count:]]
    return vectors[:, selected].T.astype(np.float32)


def _csp_features(trials: np.ndarray, filters: np.ndarray) -> np.ndarray:
    filtered = _bandpass(trials)
    projected = np.einsum("kc,nct->nkt", filters, filtered)
    variance = np.var(projected, axis=-1) + 1e-8
    return np.log(variance / variance.sum(axis=1, keepdims=True))


def headroom_fold(fold: int) -> dict[str, Any]:
    if fold not in range(5): raise ValueError(fold)
    mapping = folds(); evaluation = [p for p, f in mapping.items() if f == fold]
    training = [p for p in range(1, 26) if p not in evaluation]
    # Deterministic fold-local robust scaling estimated from outer-training sessions 01--03.
    scale_rows = []
    for participant in training:
        eeg, _, sessions = _load_subject(participant)
        for session in (1, 2, 3): scale_rows.append(eeg[sessions == session][:16, :, ::4])
    scale_array = np.concatenate(scale_rows, axis=0)
    center = np.median(scale_array, axis=(0, 2))[:, None]
    scale = np.median(np.abs(scale_array - center[None]), axis=(0, 2))[:, None] / .67448975
    scale = np.maximum(scale, np.median(scale[scale > 0]) * 1e-3)

    day1: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for participant in range(1, 26):
        eeg, labels, sessions = _load_subject(participant)
        trials = ((eeg[sessions == 1] - center[None]) / scale[None]).astype(np.float32)
        trial_labels = labels[sessions == 1]
        day1[participant] = (trials, trial_labels)
    # The number of support trials is frozen using Day-1 support only.  Every
    # participant/context contributes exactly the same count within each class.
    equal_trials_per_class = {
        label: min(int(np.sum(day1[p][1] == label)) for p in range(1, 26))
        for label in (0, 1)
    }
    phenotype: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for participant in range(1, 26):
        trials, trial_labels = day1[participant]
        for label in (0, 1):
            available = np.flatnonzero(trial_labels == label)
            count = equal_trials_per_class[label]
            selected = available[np.round(np.linspace(0, len(available) - 1, count)).astype(int)]
            phenotype[(participant, label)] = _phenotype(trials[selected])
    population = {
        label: _population_phenotype([phenotype[(participant, label)] for participant in training])
        for label in (0, 1)
    }

    with np.load(DERIVED / "frozen_masks.npz", allow_pickle=False) as data:
        masks = np.asarray(data["masks"], bool); families = data["families"].astype(str)
    primary_indices = {family: np.flatnonzero(families == family) for family in FAMILIES}
    rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    for recipient in evaluation:
        train_trials, train_labels = day1[recipient]
        filters = _csp_filters(train_trials[train_labels == 0], train_trials[train_labels == 1])
        decoder = LinearDiscriminantAnalysis().fit(_csp_features(train_trials, filters), train_labels)
        eeg, labels, sessions = _load_subject(recipient)
        wrong_donors = [p for p in evaluation if p != recipient]
        outputs_for_decoder: dict[str, list[np.ndarray]] = defaultdict(list)
        reconstruction_cache: dict[Any, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]] = {}
        decoder_labels: list[int] = []
        decoder_days: list[int] = []
        for day in (2, 3):
            session_trials = ((eeg[sessions == day] - center[None]) / scale[None]).astype(np.float32)
            session_labels = labels[sessions == day]
            for trial_index, (clean, label) in enumerate(zip(session_trials, session_labels)):
                decoder_labels.append(int(label)); decoder_days.append(day)
                trial_outputs: dict[str, list[np.ndarray]] = defaultdict(list)
                for family_index, family in enumerate(FAMILIES):
                    indices = primary_indices[family]
                    mask_index = int(indices[(recipient * 1009 + day * 101 + trial_index * 7 + family_index) % len(indices)])
                    mask = masks[mask_index]
                    contexts: dict[str, np.ndarray] = {
                        "POP": np.asarray(population[int(label)]["covariance"]),
                        "MATCH": np.asarray(phenotype[(recipient, int(label))]["covariance"]),
                    }
                    for donor in wrong_donors:
                        contexts[f"WRONG-{donor:02d}"] = np.asarray(phenotype[(donor, int(label))]["covariance"])
                    for method, covariance in contexts.items():
                        restored = _reconstruct(
                            clean, mask, covariance, operator_cache=reconstruction_cache,
                            cache_key=(int(label), mask_index, method),
                        )
                        metrics = _masked_metrics(clean, restored, mask)
                        rows.append({
                            "fold": fold, "participant": recipient, "day": day,
                            "class": int(label), "trial": trial_index, "family": family,
                            "mask_id": mask_index, "method": method,
                            "donor": int(method.split("-")[1]) if method.startswith("WRONG-") else "",
                            **metrics,
                        })
                        trial_outputs[method].append(restored)
                for method, values in trial_outputs.items(): outputs_for_decoder[method].append(np.mean(values, axis=0))
        y_true = np.asarray(decoder_labels)
        for method, trials in outputs_for_decoder.items():
            prediction = decoder.predict(_csp_features(np.asarray(trials), filters))
            for day in (2, 3):
                selected = np.asarray(decoder_days) == day
                decoder_rows.append({
                    "fold": fold, "participant": recipient, "day": day, "method": method,
                    "mi_accuracy": float(np.mean(prediction[selected] == y_true[selected])),
                })
    destination = RESULT / "j1" / f"fold_{fold:02d}"
    _csv(destination / "unit_metrics.csv", rows); _csv(destination / "decoder_metrics.csv", decoder_rows)
    _json(destination / "result_summary.json", {
        "fold": fold, "training": training, "evaluation": evaluation, "unit_rows": len(rows),
        "participants": len(evaluation), "equal_trials_per_class": equal_trials_per_class,
        "day_4_5_opened": False,
    })
    return {"fold": fold, "unit_rows": len(rows), "participants": len(evaluation)}


def _exact_sign_flip_p(values: np.ndarray) -> float:
    """Exact one-sided randomization p for the participant mean (n <= 25)."""
    values = np.asarray(values, dtype=np.float64)
    split = len(values) // 2
    first, second = values[:split], values[split:]
    sums_first = np.asarray([sum((1 if bits >> i & 1 else -1) * first[i] for i in range(len(first))) for bits in range(1 << len(first))])
    sums_second = np.sort(np.asarray([sum((1 if bits >> i & 1 else -1) * second[i] for i in range(len(second))) for bits in range(1 << len(second))]))
    threshold = float(values.sum())
    count = sum(len(sums_second) - int(np.searchsorted(sums_second, threshold - value, side="left")) for value in sums_first)
    return float(count / (2 ** len(values)))


def aggregate() -> dict[str, Any]:
    rows: list[dict[str, str]] = []; decoder_rows: list[dict[str, str]] = []
    for fold in range(5):
        rows.extend(csv.DictReader((RESULT / "j1" / f"fold_{fold:02d}" / "unit_metrics.csv").open()))
        decoder_rows.extend(csv.DictReader((RESULT / "j1" / f"fold_{fold:02d}" / "decoder_metrics.csv").open()))
    # Canonical aggregation: trial/mask -> class -> day -> participant, with
    # the five preregistered mask families equal-weighted inside each cell.
    cell_rows: list[dict[str, Any]] = []
    for participant in range(1, 26):
        for day in (2, 3):
            for label in (0, 1):
                for family in FAMILIES:
                    selected = [r for r in rows if int(r["participant"]) == participant and int(r["day"]) == day and int(r["class"]) == label and r["family"] == family]
                    by_method: dict[str, list[float]] = defaultdict(list)
                    by_erd: dict[str, list[float]] = defaultdict(list)
                    for row in selected:
                        by_method[row["method"]].append(float(row["rrmse"]))
                        by_erd[row["method"]].append(float(row["erd_distortion"]))
                    match, pop = np.mean(by_method["MATCH"]), np.mean(by_method["POP"])
                    wrong = np.mean([np.mean(v) for k, v in by_method.items() if k.startswith("WRONG-")])
                    cell_rows.append({
                        "participant": participant, "day": day, "class": label, "family": family,
                        "match_rrmse": match, "pop_rrmse": pop, "mean_wrong_rrmse": wrong,
                        "H_P": pop-match, "H_W": wrong-match,
                        "erd_margin": np.mean(by_erd["POP"])-np.mean(by_erd["MATCH"]),
                    })
    participant_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for participant in range(1, 26):
        participant_cells = [r for r in cell_rows if r["participant"] == participant]
        for family in FAMILIES:
            subset = [r for r in participant_cells if r["family"] == family]
            family_rows.append({"participant": participant, "family": family, "H_P": np.mean([r["H_P"] for r in subset]), "H_W": np.mean([r["H_W"] for r in subset])})
        match = np.mean([r["match_rrmse"] for r in participant_cells])
        pop = np.mean([r["pop_rrmse"] for r in participant_cells])
        wrong = np.mean([r["mean_wrong_rrmse"] for r in participant_cells])
        # Decoder non-inferiority is MATCH minus POP, higher is better.
        dec = [r for r in decoder_rows if int(r["participant"]) == participant]
        match_dec = np.mean([float(r["mi_accuracy"]) for r in dec if r["method"] == "MATCH"])
        pop_dec = np.mean([float(r["mi_accuracy"]) for r in dec if r["method"] == "POP"])
        participant_rows.append({
            "participant": participant, "fold": folds()[participant],
            "H_P": pop-match, "H_W": wrong-match,
            "match_rrmse": match, "pop_rrmse": pop, "mean_wrong_rrmse": wrong,
            "mi_accuracy_match": match_dec, "mi_accuracy_pop": pop_dec,
            "mi_accuracy_margin": match_dec-pop_dec,
        })
    _csv(RESULT / "j1" / "participant_effects.csv", participant_rows)
    _csv(RESULT / "j1" / "participant_family_effects.csv", family_rows)
    effects: dict[str, dict[str, Any]] = {}
    for key in ("H_P", "H_W"):
        values = np.asarray([row[key] for row in participant_rows])
        positive = int(np.sum(values > 0))
        effects[key] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "positive": positive, "n": 25, "one_sided_exact_p": _exact_sign_flip_p(values),
            "values": values.tolist(),
        }
    day_class_rows: list[dict[str, Any]] = []
    for participant in range(1, 26):
        for day in (2, 3):
            for label in (0, 1):
                selected = [r for r in cell_rows if r["participant"] == participant and r["day"] == day and r["class"] == label]
                day_class_rows.append({"participant": participant, "day": day, "class": label, "H_P": np.mean([r["H_P"] for r in selected]), "H_W": np.mean([r["H_W"] for r in selected])})
    _csv(RESULT / "j1" / "participant_day_class_effects.csv", day_class_rows)
    day_effects: dict[str, dict[str, float]] = {}
    for day in (2, 3):
        day_effects[str(day)] = {key: float(np.mean([r[key] for r in day_class_rows if r["day"] == day])) for key in ("H_P", "H_W")}
    class_effects: dict[str, dict[str, float]] = {}
    for label in (0, 1):
        class_effects[str(label)] = {key: float(np.mean([r[key] for r in day_class_rows if r["class"] == label])) for key in ("H_P", "H_W")}
    family_means = {
        family: {key: float(np.mean([r[key] for r in family_rows if r["family"] == family])) for key in ("H_P", "H_W")}
        for family in FAMILIES
    }
    decoder_mean = float(np.mean([row["mi_accuracy_margin"] for row in participant_rows]))
    erd_margin = float(np.mean([r["erd_margin"] for r in cell_rows]))
    go = (
        all(effects[k]["mean"] >= .005 and effects[k]["median"] > 0 and effects[k]["positive"] >= 18 and effects[k]["one_sided_exact_p"] < .05 for k in ("H_P", "H_W"))
        and all(v[key] > 0 for v in day_effects.values() for key in ("H_P", "H_W"))
        and all(v[key] >= -.02 for v in class_effects.values() for key in ("H_P", "H_W"))
        and erd_margin >= -.02 and decoder_mean >= -.02
    )
    decision = "TASK_PHENOTYPE_HEADROOM_GO" if go else "TASK_PHENOTYPE_HEADROOM_NO_GO"
    route = {
        "decision": decision, "gpu_model_authorized": go, "effects": effects,
        "day_effects": day_effects, "class_effects": class_effects,
        "family_effects": family_means, "mean_mi_accuracy_margin": decoder_mean,
        "mean_erd_distortion_margin": erd_margin,
        "participant_denominator": 25, "day_4_5_opened": False,
        "terminal_boundary": "Failure permanently stops subject-specific EEG denoising experiments.",
    }
    _json(RESULT / "routing_decision.json", route)
    _json(RESULT / "result_summary.json", {
        "stage": "J1_HEADROOM", "decision": decision, "participants": 25,
        "sessions_evaluated": [2, 3], "session_04_05_opened": False, "effects": effects,
    })
    return route


def report() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    route = json.loads((RESULT / "routing_decision.json").read_text())
    participants = list(csv.DictReader((RESULT / "j1" / "participant_effects.csv").open()))
    masks = list(csv.DictReader((RESULT / "frozen" / "mask_manifest.csv").open()))
    figure_root = RESULT / "figures"; figure_root.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, 26); hp = np.asarray([float(r["H_P"]) for r in participants]); hw = np.asarray([float(r["H_W"]) for r in participants])
    fig, axis = plt.subplots(figsize=(9, 4.5)); axis.axhline(0, color="black", lw=.8)
    axis.plot(x, hp, "o-", label="MATCH vs POP"); axis.plot(x, hw, "s-", label="MATCH vs mean WRONG")
    axis.set(xlabel="Participant", ylabel="RRMSE utility (positive is MATCH better)", xticks=x)
    axis.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_root / "participant_headroom.png", dpi=180); plt.close(fig)

    lines = [
        "# SHU task-phenotype subject-aware restoration exploration", "",
        f"Final routing decision: `{route['decision']}`.", "",
        "## Data and protocol validity", "",
        "The datalake contains 25 participants × five sessions and 11,988 real processed MI trials. The available asset is a 256 Hz trial-level LMDB derivative; it was deterministically resampled to the frozen nominal 250 Hz/1000-sample protocol. The source EDF/MAT containers are not directly present, so EDF–MAT file-level correspondence could not be independently re-audited. Trial keys, sessions, labels, shape, finiteness, and 25×5 coverage were audited. Day-4/5 payloads were never deserialized.", "",
        "PhysioMotion contributed only channel-count/duration/adjacency mask geometry from its already-opened development-20 annotations. No PhysioMotion waveform or sealed participant was read. The mapped library contains " + str(len(masks)) + " masks including the fixed electrode-dropout controls; the scientific J1 used the five frozen nonocular artifact families.", "",
        "## Day-1 task-phenotype headroom", "",
        "Class-conditional Day-1 support used equal trial counts across every MATCH, POP, and WRONG operator. POP was an equal-participant outer-fold mean; WRONG donors were the other four unseen participants in the same held-out fold. Day-2/3 query results were aggregated trial/mask → class → day → participant (n=25).", "",
        f"H_P mean/median was {route['effects']['H_P']['mean']:+.6f}/{route['effects']['H_P']['median']:+.6f}, with {route['effects']['H_P']['positive']}/25 positive and one-sided exact p={route['effects']['H_P']['one_sided_exact_p']:.6f}. H_W was {route['effects']['H_W']['mean']:+.6f}/{route['effects']['H_W']['median']:+.6f}, with {route['effects']['H_W']['positive']}/25 positive and p={route['effects']['H_W']['one_sided_exact_p']:.6f}.", "",
        f"Day effects: {route['day_effects']}. Class effects: {route['class_effects']}. Mean ERD-distortion margin versus POP was {route['mean_erd_distortion_margin']:+.6f}; frozen Day-1 CSP-LDA accuracy margin was {route['mean_mi_accuracy_margin']:+.6f}.", "",
        "## Routing boundary", "",
    ]
    if route["gpu_model_authorized"]:
        lines += ["The CPU headroom gate passed, authorizing the single frozen task-phenotype masked DET/DIFF implementation. Day-4/5 remain sealed until the later three-seed gate."]
    else:
        lines += ["The preregistered task-phenotype headroom gate failed. No DET/diffusion model training, scientific GPU screen, extra seeds, or Day-4/5 evaluation was run. Under the terminal instruction, subject-specific EEG denoising experimentation stops here; this is a failure of this fixed task-phenotype representation/probe, not a family-wide mathematical claim about diffusion or personalization."]
    Path("reports/shu_task_phenotype_diffusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "REPORT_COMPLETE", "decision": route["decision"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("materialize"); p.add_argument("--subject", type=int, required=True)
    p = sub.add_parser("headroom-fold"); p.add_argument("--fold", type=int, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.stage == "freeze": output = freeze()
    elif args.stage == "materialize": output = materialize(args.subject)
    elif args.stage == "headroom-fold": output = headroom_fold(args.fold)
    elif args.stage == "aggregate": output = aggregate()
    else: output = report()
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__": main()
