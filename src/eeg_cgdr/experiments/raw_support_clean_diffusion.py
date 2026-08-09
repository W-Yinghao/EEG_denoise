"""Raw temporal neural support conditioned clean-EEG diffusion experiment.

Scientific aggregation is protocol -> seed -> participant.  Model-side stages
never open evaluator arrays.  Evaluator-only geometry/oracle stages are named
explicitly and never create deployable outputs.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_subject_diffusion_next as nxt
from eeg_cgdr.experiments import diffusion_fair_neural_prior as fair

SAME = ("same_01", "same_02", "same_03")
SESSIONS = ("01T", "02T", "03T")
TRAIN_SESSIONS = ("01T", "02T", "03T", "04E", "05E")


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _root(c: Mapping[str, Any], key: str) -> Path:
    return Path(str(c[key]))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seed_fold(root: Path, seed: int, fold: int) -> Path:
    return root / "seeds" / str(seed) / "folds" / f"fold_{fold:02d}"


def _bandpass(value: np.ndarray, sfreq: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, (1, 45), btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, value, axis=-1).astype(np.float64)


def _robust_correlation_geometry(value: np.ndarray) -> np.ndarray:
    center = np.median(value, axis=1, keepdims=True)
    scale = np.median(np.abs(value - center), axis=1, keepdims=True) / .67448975
    z = np.clip((value - center) / np.maximum(scale, 1e-12), -8, 8)
    cov = np.cov(z)
    target = np.trace(cov) / len(cov)
    return (.9 * cov + .1 * target * np.eye(len(cov))).astype(np.float64)


def _spd_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, float))
    return (vectors * np.maximum(values, 1e-10) ** power) @ vectors.T


def _spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, float))
    return (vectors * np.log(np.maximum(values, 1e-10))) @ vectors.T


def _spd_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, float))
    return (vectors * np.exp(values)) @ vectors.T


def _karcher_mean(matrices: list[np.ndarray], tolerance: float = 1e-9, maximum_iterations: int = 100) -> np.ndarray:
    """Affine-invariant Karcher barycenter with participant-equal weights."""
    if not matrices:
        raise ValueError("empty SPD population")
    current = _spd_exp(np.mean([_spd_log(x) for x in matrices], axis=0))
    for _ in range(maximum_iterations):
        root = _spd_power(current, .5); inverse = _spd_power(current, -.5)
        tangent = np.mean([_spd_log(inverse @ x @ inverse) for x in matrices], axis=0)
        if np.linalg.norm(tangent, "fro") < tolerance:
            break
        current = root @ _spd_exp(tangent) @ root
        current = .5 * (current + current.T)
    return current


def _log_barycenter(matrices: list[np.ndarray]) -> np.ndarray:
    return _spd_exp(np.mean([_spd_log(x) for x in matrices], axis=0))


def _airm(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.linalg import eigvalsh
    return float(np.linalg.norm(np.log(np.maximum(eigvalsh(right, left), 1e-10))))


def _log_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(_spd_log(left) - _spd_log(right), "fro"))


def _psd_profile(value: np.ndarray, sfreq: float) -> np.ndarray:
    from scipy.signal import welch
    freq, psd = welch(value, fs=sfreq, nperseg=min(500, value.shape[-1]), axis=-1)
    keep = (freq >= 1) & (freq <= 45)
    return np.log(np.maximum(psd[:, keep], 1e-20)).mean(1)


def _low_eog_patches(c: Mapping[str, Any], participant: int, session: str, *,
                     require_120: bool, count: int | None = None,
                     location: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    eeg, eog, sfreq, events = v11._load_session(c, participant, session)
    support, _ = v11._support_query_ranges(events, eeg.shape[1], sfreq)
    available = (support.stop - support.start) / sfreq
    stop = min(support.stop, support.start + int(round(float(c["support_seconds"]) * sfreq)))
    length = int(round(float(c["support_patch_seconds"]) * sfreq))
    starts = list(range(support.start, stop - length + 1, length))
    energies = np.asarray([np.sqrt(np.mean(eog[:, start:start + length].astype(float) ** 2)) for start in starts])
    if not starts or (require_120 and available < float(c["support_seconds"])):
        return np.empty((0, eeg.shape[0], length), np.float32), {"available_seconds": available, "candidate_windows": len(starts), "selected_windows": 0, "eligible": 0}
    threshold = float(np.quantile(energies, float(c["support_low_eog_quantile"])))
    selected_starts = [start for start, energy in zip(starts, energies) if energy <= threshold]
    if count is not None:
        if len(selected_starts) < count:
            return np.empty((0, eeg.shape[0], length), np.float32), {"available_seconds": available, "candidate_windows": len(starts), "low_eog_windows": len(selected_starts), "selected_windows": 0, "eligible": 0}
        index = np.linspace(0, len(selected_starts) - 1, count).round().astype(int)
        selected_starts = [selected_starts[i] for i in index]
    patches = np.stack([eeg[:, start:start + length] for start in selected_starts]).astype(np.float32)
    if location is not None and scale is not None:
        patches = ((patches - location[None, :, None]) / scale[None, :, None]).astype(np.float32)
    return patches, {"available_seconds": available, "candidate_windows": len(starts), "low_eog_windows": int(np.sum(energies <= threshold)), "selected_windows": len(patches), "eligible": int(len(patches) > 0), "threshold": threshold}


def _later_low_eog_geometry(c: Mapping[str, Any], participant: int, session: str) -> tuple[np.ndarray, np.ndarray, int]:
    eeg, eog, sfreq, events = v11._load_session(c, participant, session)
    _, query = v11._support_query_ranges(events, eeg.shape[1], sfreq)
    length = int(round(float(c["support_patch_seconds"]) * sfreq))
    starts = list(range(query.start, query.stop - length + 1, length))
    energy = np.asarray([np.sqrt(np.mean(eog[:, start:start + length].astype(float) ** 2)) for start in starts])
    threshold = np.quantile(energy, float(c["query_low_eog_quantile"]))
    patches = [eeg[:, start:start + length] for start, value in zip(starts, energy) if value <= threshold]
    filtered = _bandpass(np.concatenate(patches, axis=1), sfreq)
    return _robust_correlation_geometry(filtered), _psd_profile(filtered, sfreq), len(patches)


def stage_audit(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    strict = _root(c, "strict_root")
    units = []
    for fold in range(9):
        units.extend(_read(strict / "prepared" / f"fold_{fold:02d}" / "unit_manifest.csv"))
    summary = {"status": "RAW_SUPPORT_PROTOCOL_FROZEN", "participants": 9,
               "eligible_protocol_units": sum(int(row["eligible_120"]) for row in units),
               "availability_denominator": len(units), "support_seconds": 120,
               "support_patches": 16, "patch_seconds": 2, "token_dimension": 64,
               "primary_seed": int(c["primary_seed"]), "DDIM_steps": 25, "K": 8,
               "scientific_aggregation": "protocol -> seed -> participant", "a_track_touched": False}
    if summary["eligible_protocol_units"] != 26 or summary["availability_denominator"] != 27:
        raise RuntimeError(summary)
    _json(_root(c, "result_root") / "frozen_protocol.json", summary); _json(run / "result_summary.json", summary)
    return summary


def stage_geometry_support(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    root = _root(c, "geometry_root") / "support"; coverage = []
    for participant in range(1, 10):
        arrays: dict[str, np.ndarray] = {}
        for session in SESSIONS:
            patches, meta = _low_eog_patches(c, participant, session, require_120=True)
            coverage.append({"participant": participant, "session": session, **meta})
            if not len(patches):
                continue
            filtered = _bandpass(np.concatenate(list(patches), axis=1), 250.0)
            halves = np.array_split(filtered, 2, axis=1)
            cov = _robust_correlation_geometry(filtered)
            arrays[f"{session}_cov"] = cov
            arrays[f"{session}_psd"] = _psd_profile(filtered, 250.0)
            arrays[f"{session}_reliability"] = np.asarray(np.corrcoef(_robust_correlation_geometry(halves[0]).ravel(), _robust_correlation_geometry(halves[1]).ravel())[0, 1])
        root.mkdir(parents=True, exist_ok=True); np.savez_compressed(root / f"participant_{participant:02d}.npz", **arrays)
    _csv(_root(c, "geometry_root") / "support_coverage.csv", coverage)
    summary = {"status": "CORRECTED_SUPPORT_GEOMETRY_EXTRACTED", "participants": 9,
               "definition": "robust spatial-correlation geometry after per-channel median/MAD",
               "support_query_low_eog_quantile": float(c["support_low_eog_quantile"]), "query_opened": False}
    _json(run / "result_summary.json", summary); return summary


def stage_geometry_evaluate(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    root = _root(c, "geometry_root"); support = {p: np.load(root / "support" / f"participant_{p:02d}.npz") for p in range(1, 10)}
    rows: list[dict[str, Any]] = []
    try:
        for participant in range(1, 10):
            for session in SESSIONS:
                key = f"{session}_cov"
                if key not in support[participant]:
                    rows.append({"participant": participant, "session": session, "eligible": 0}); continue
                cq, pq, windows = _later_low_eog_geometry(c, participant, session)
                owners = [p for p in range(1, 10) if p != participant and key in support[p]]
                covs = [np.asarray(support[p][key]) for p in owners]
                c_airm = _karcher_mean(covs); c_log = _log_barycenter(covs)
                cs = np.asarray(support[participant][key]); ps = np.asarray(support[participant][f"{session}_psd"])
                wrong_airm = [_airm(cq, value) for value in covs]; wrong_log = [_log_distance(cq, value) for value in covs]
                wrong_psd = [np.linalg.norm(pq - np.asarray(support[p][f"{session}_psd"])) for p in owners]
                p0 = np.mean([np.asarray(support[p][f"{session}_psd"]) for p in owners], axis=0)
                rows.append({"participant": participant, "session": session, "eligible": 1, "query_windows": windows,
                             "population_owners": ";".join(map(str, owners)), "H_P_airm": _airm(cq, c_airm) - _airm(cq, cs),
                             "H_W_airm": float(np.mean(wrong_airm)) - _airm(cq, cs),
                             "H_P_logeuclidean": _log_distance(cq, c_log) - _log_distance(cq, cs),
                             "H_W_logeuclidean": float(np.mean(wrong_log)) - _log_distance(cq, cs),
                             "H_P_psd": float(np.linalg.norm(pq - p0) - np.linalg.norm(pq - ps)),
                             "H_W_psd": float(np.mean(wrong_psd) - np.linalg.norm(pq - ps)),
                             "support_reliability": float(support[participant][f"{session}_reliability"]), "query_opened_evaluator_only": 1})
    finally:
        for value in support.values(): value.close()
    _csv(root / "geometry_session_metrics.csv", rows); summary = {"status": "CORRECTED_GEOMETRY_EVALUATED", "eligible_rows": sum(int(r["eligible"]) for r in rows), "denominator": 27}
    _json(run / "result_summary.json", summary); return summary


def _sign_flip(values: np.ndarray, one_sided: bool = False) -> float:
    signs = ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1) * 2 - 1
    null = (signs * values[None]).mean(1); observed = float(values.mean())
    return float(np.mean(null >= observed)) if one_sided else float(np.mean(np.abs(null) >= abs(observed)))


def _summary(c: Mapping[str, Any], values: np.ndarray, seed_values: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    values = np.asarray(values, float); rng = np.random.default_rng(int(c["bootstrap_seed"])); indices = rng.integers(0, len(values), size=(int(c["bootstrap_replicates"]), len(values))); rep = values[indices].mean(1)
    result = {"mean": float(values.mean()), "median": float(np.median(values)), "positive": int(np.sum(values > 0)), "participant_values": values.tolist(), "one_sided_exact_sign_flip": _sign_flip(values, True), "two_sided_exact_sign_flip": _sign_flip(values), "descriptive_ci": [float(np.quantile(rep, .025)), float(np.quantile(rep, .975))], "leave_one_participant_out": [float(np.delete(values, i).mean()) for i in range(len(values))]}
    if seed_values is not None: result["seed_means"] = {str(seed): float(value.mean()) for seed, value in seed_values.items()}
    return result


def stage_geometry_aggregate(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    root = _root(c, "geometry_root"); rows = _read(root / "geometry_session_metrics.csv"); participant = []
    keys = ("H_P_airm", "H_W_airm", "H_P_logeuclidean", "H_W_logeuclidean", "H_P_psd", "H_W_psd", "support_reliability")
    for p in range(1, 10):
        take = [r for r in rows if int(r["participant"]) == p and int(r["eligible"])]
        participant.append({"participant": p, **{key: float(np.mean([float(r[key]) for r in take])) for key in keys}})
    _csv(root / "corrected_geometry_participant_first.csv", participant)
    metrics = {key: _summary(c, np.asarray([r[key] for r in participant])) for key in keys}
    summary = {"status": "ROBUST_SPATIAL_CORRELATION_GEOMETRY_REAUDITED", "metrics": metrics, "scientific_n": 9, "aggregation": "session -> participant", "population_airm": "Karcher affine-invariant barycenter", "population_logeuclidean": "log-domain barycenter", "historical_results_overwritten": False}
    _json(root / "corrected_geometry_summary.json", summary)
    figdir = root / "figures"; figdir.mkdir(parents=True, exist_ok=True); fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(1, 10); ax.axhline(0, color="black", lw=.8); ax.plot(x, [r["H_P_airm"] for r in participant], "o-", label="MATCH−POP"); ax.plot(x, [r["H_W_airm"] for r in participant], "s-", label="MATCH−WRONG"); ax.set(xlabel="Participant", ylabel="Positive geometry headroom"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "corrected_geometry.png", dpi=180); plt.close(fig)
    _json(run / "result_summary.json", summary); return summary


def _load_old_aligned(c: Mapping[str, Any], fold: int, device: Any):
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic, DeterministicEOGResidual, EOGResidualConfig, EOGResidualDiffusion
    root = _root(c, "previous_root") / "clean_neural_aligned_diffusion" / "models" / str(c["primary_seed"]) / f"fold_{fold:02d}"
    cp = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False); cfg = EOGResidualConfig(**cp["config"])
    det1 = DeterministicEOGResidual(cfg).to(device); det2 = CapacityMatchedDeterministic(cfg).to(device); diff = EOGResidualDiffusion(cfg).to(device)
    det1.load_state_dict(cp["det1"]); det2.load_state_dict(cp["det2"]); diff.load_state_dict(cp["diff"]); det1.eval(); det2.eval(); diff.eval()
    return cp, det1, det2, diff


def stage_oracle_action(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    """Evaluator-only covariance actionability; outputs metrics, never deployable arrays."""
    import torch
    fold = task_index; recipient = fold + 1; seed = int(c["primary_seed"]); device = torch.device("cuda")
    cp, det1, det2, diff = _load_old_aligned(c, fold, device); rescale = np.asarray(cp["residual_scale"], np.float32)
    supportroot = _root(c, "geometry_root") / "support"; support = {p: np.load(supportroot / f"participant_{p:02d}.npz") for p in range(1, 10)}; rows = []
    base = _seed_fold(_root(c, "strict_root"), seed, fold)
    try:
        for unit_index, protocol in enumerate(SAME):
            inf = np.load(base / "units" / protocol / "inference.npz"); ev = np.load(base / "units" / protocol / "evaluator.npz")
            if not int(inf["recipient_eligible"]): continue
            session = f"{unit_index + 1:02d}T"; key = f"{session}_cov"; owners = [p for p in range(1, 10) if p != recipient and key in support[p]]
            pop_cov = _karcher_mean([np.asarray(support[p][key]) for p in owners]); match_cov = np.asarray(support[recipient][key]); wrong = [(p, np.asarray(support[p][key])) for p in owners]
            y = np.asarray(inf["paired_y"], np.float32); eog = np.asarray(inf["paired_eog"], np.float32); xphysical = np.asarray(ev["paired_x"], np.float32)
            loc, scale = np.asarray(inf["eeg_location"]), np.asarray(inf["eeg_scale"]); x = ((xphysical - loc[None, :, None]) / scale[None, :, None]).astype(np.float32); oracle_cov = _robust_correlation_geometry(_bandpass(np.concatenate([window[:, :500] for window in x], axis=1), 250.0))
            contexts = [("POP", pop_cov), ("MATCH", match_cov), ("ORACLE", oracle_cov)] + [(f"WRONG-{p}", cov) for p, cov in wrong]
            hpop = np.asarray(inf["h_pop"]); a0native = v11.apply_transfer(hpop, eog); bank = v11._noise_bank(y.shape, seed + fold * 100000 + unit_index * 10000, 8)
            for name, cov in contexts:
                matrix = fair._invsqrt(cov); inverse = np.linalg.inv(matrix); yz = fair._align(y, matrix); a0z = fair._align(a0native, matrix); yt = torch.as_tensor(yz, device=device); et = torch.as_tensor(eog, device=device); at = torch.as_tensor(a0z, device=device)
                with torch.no_grad():
                    rd = det1(y=yt, eog=et, a0=at); d2 = det2(y=yt, eog=et, a0=at, r_det=rd)
                    samples = [diff.sample(y=yt, eog=et, a0=at, r_det=rd, initial_noise=torch.as_tensor(noise, device=device)) for noise in bank]
                predictions = {"DET2": rd + d2, "DIFF25-K1": rd + samples[0], "DIFF25-K8": rd + torch.stack(samples).mean(0)}
                for estimator, residual in predictions.items():
                    correction = fair._align(a0z + residual.cpu().numpy() * rescale[None, :, None], inverse); correction[..., 500:] = 0
                    restored = v11.gamma_correction(y, correction, float(inf["gamma"])); physical = restored * scale[None, :, None] + loc[None, :, None]
                    rows.append({"participant": recipient, "protocol": protocol, "context": name, "estimator": estimator, "rrmse": v11.rrmse(physical[..., :500], xphysical[..., :500]), "evaluator_only_oracle": int(name == "ORACLE")})
    finally:
        for item in support.values(): item.close()
    _csv(_root(c, "geometry_root") / "oracle_tasks" / f"fold_{fold:02d}.csv", rows); summary = {"status": "EVALUATOR_ONLY_ORACLE_ACTIONABILITY_COMPLETED", "fold": fold, "rows": len(rows)}; _json(run / "result_summary.json", summary); return summary


def stage_oracle_aggregate(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    root = _root(c, "geometry_root"); rows = []
    for path in sorted((root / "oracle_tasks").glob("fold_*.csv")): rows.extend(_read(path))
    _csv(root / "oracle_actionability_metrics.csv", rows); participant = []
    for p in range(1, 10):
        take = [r for r in rows if int(r["participant"]) == p]; by = defaultdict(list)
        for row in take: by[(row["estimator"], row["context"])].append(float(row["rrmse"]))
        wrong = {est: [np.mean(values) for (model, context), values in by.items() if model == est and context.startswith("WRONG-")] for est in ("DET2", "DIFF25-K1", "DIFF25-K8")}
        item: dict[str, Any] = {"participant": p}
        for estimator in ("DET2", "DIFF25-K1", "DIFF25-K8"):
            pop = np.mean(by[(estimator, "POP")]); oracle = np.mean(by[(estimator, "ORACLE")]); item[f"U_O_{estimator}"] = pop - oracle; item[f"U_OW_{estimator}"] = float(np.mean(wrong[estimator])) - oracle
        item["DeltaO_K8"] = item["U_O_DIFF25-K8"] - item["U_O_DET2"]; participant.append(item)
    _csv(root / "oracle_actionability_participant_first.csv", participant); effects = {key: _summary(c, np.asarray([row[key] for row in participant])) for key in participant[0] if key != "participant"}
    primary = effects["U_O_DIFF25-K8"]; actionable = primary["mean"] >= .005 and primary["median"] > 0 and primary["positive"] >= 7
    route = {"status": "COVARIANCE_ACTIONABILITY_CEILING_PRESENT" if actionable else "COVARIANCE_ACTIONABILITY_ROUTE_CLOSED", "oracle_is_evaluator_only": True, "whitening_model_retraining_forbidden": True, "development_only": True}
    summary = {"effects": effects, "routing": route}; _json(root / "oracle_actionability_summary.json", summary); _json(root / "oracle_actionability_route.json", route)
    _json(run / "result_summary.json", summary); return summary


def stage_materialize(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    fold = task_index; recipient = fold + 1; strict = _root(c, "strict_root"); base = strict / "prepared" / f"fold_{fold:02d}"; arrays: dict[str, np.ndarray] = {}; rows = []
    training = [p for p in range(1, 10) if p != recipient]
    # The frozen scientific panel uses sessions 01--03, while the unchanged
    # outer-training pseudo-pair table also contains sessions 04/05.  All five
    # therefore need episode support contexts; evaluation remains 01--03 only.
    reference = np.load(base / "units" / SAME[0] / "inference.npz")
    fold_location, fold_scale = np.asarray(reference["eeg_location"]), np.asarray(reference["eeg_scale"])
    for session_index, session in enumerate(TRAIN_SESSIONS):
        loc, scale = fold_location, fold_scale
        owner_two = []
        for p in range(1, 10):
            patches, meta = _low_eog_patches(c, p, session, require_120=True, count=int(c["support_patches"]), location=loc, scale=scale)
            rows.append({"fold": fold, "recipient": recipient, "participant": p, "session": session, "role": "MATCH", **meta})
            if len(patches): arrays[f"p{p}_s{session_index + 1}"] = patches
            if p in training:
                pair, pmeta = _low_eog_patches(c, p, session, require_120=False, count=2, location=loc, scale=scale)
                if len(pair) != 2: raise RuntimeError(f"population owner {p}/{session} lacks two patches")
                owner_two.append(pair); rows.append({"fold": fold, "recipient": recipient, "participant": p, "session": session, "role": "POP_CONTRIBUTION", **pmeta})
        arrays[f"pop_s{session_index + 1}"] = np.concatenate(owner_two, axis=0)
        if arrays[f"pop_s{session_index + 1}"].shape != (16, 3, 500): raise RuntimeError("population set shape/fairness violation")
    target = _root(c, "model_root") / "materialized" / f"fold_{fold:02d}"; target.mkdir(parents=True, exist_ok=True); np.savez_compressed(target / "support_sets.npz", **arrays); _csv(target / "support_manifest.csv", rows)
    summary = {"status": "RAW_SUPPORT_CONTEXTS_MATERIALIZED", "fold": fold, "recipient": recipient, "population_owners": training, "contexts_equal_token_count": True, "query_opened": False}; _json(run / "result_summary.json", summary); return summary


def _support_batch(contexts: Mapping[str, np.ndarray], subjects: np.ndarray, sessions: np.ndarray, use_match: np.ndarray) -> np.ndarray:
    result = []
    for subject, session, match in zip(subjects, sessions, use_match):
        key = f"p{int(subject)}_s{int(session)}"
        result.append(np.asarray(contexts[key] if match and key in contexts else contexts[f"pop_s{int(session)}"], np.float32))
    return np.stack(result)


def _train(c: Mapping[str, Any], seed: int, fold: int, device: Any, *, technical: bool) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from eeg_cgdr.models.raw_support_clean_diffusion import DeterministicRawSupportCleaner, EMA, RawSupportCleanConfig, RawSupportCleanDiffusion, checkpoint_payload
    strict = _seed_fold(_root(c, "strict_root"), int(c["primary_seed"]), fold)
    with np.load(strict / "training_pairs.npz") as data:
        y_all = np.asarray(data["y"], np.float32); clean_all = y_all - np.asarray(data["a"], np.float32); subjects_all = np.asarray(data["subject"], int); sessions_all = np.asarray(data["session"], int)
    contexts = np.load(_root(c, "model_root") / "materialized" / f"fold_{fold:02d}" / "support_sets.npz")
    if technical:
        train_index = np.arange(min(16, len(y_all))); validation_index = np.arange(min(16, len(y_all)), min(32, len(y_all)))
    else:
        train_index = np.arange(len(y_all)); validation_index = np.asarray([], int)
    torch.manual_seed(seed + fold); torch.cuda.manual_seed_all(seed + fold); np.random.seed(seed + fold)
    cfg = RawSupportCleanConfig(); det = DeterministicRawSupportCleaner(cfg).to(device); diff = RawSupportCleanDiffusion(cfg).to(device); diff.backbone.load_state_dict(det.backbone.state_dict())
    pdet = sum(p.numel() for p in det.parameters()); pdiff = sum(p.numel() for p in diff.parameters())
    if pdet != pdiff: raise RuntimeError("DET/DIFF parameter mismatch")
    updates = int(c["technical_updates"] if technical else c["training_updates"]); batch = int(c["batch_size"]); rng = np.random.default_rng(seed + fold); schedule = []
    for _ in range(updates):
        index = train_index if technical else rng.choice(train_index, size=batch, replace=True)
        match = np.ones(len(index), bool) if technical else rng.random(len(index)) < .5
        schedule.append((index, match))
    opt_det = AdamW(det.parameters(), lr=float(c["learning_rate"])); opt_diff = AdamW(diff.parameters(), lr=float(c["learning_rate"])); ema_det = EMA(det, float(c["ema_decay"])); ema_diff = EMA(diff, float(c["ema_decay"])); generator = torch.Generator(device=device).manual_seed(seed + fold + 999); curve = []; started = time.monotonic()
    def fields(index: np.ndarray, match: np.ndarray):
        support = _support_batch(contexts, subjects_all[index], sessions_all[index], match)
        return torch.as_tensor(y_all[index], device=device), torch.as_tensor(clean_all[index], device=device), torch.as_tensor(support, device=device)
    fixed_y, fixed_clean, fixed_support = fields(train_index, np.ones(len(train_index), bool)); fixed_t = torch.full((len(train_index),), 500, device=device, dtype=torch.long); fixed_noise = torch.randn(fixed_clean.shape, device=device, generator=generator)
    det.eval(); diff.eval()
    with torch.no_grad(): initial_det = float(((det(query_y=fixed_y, support_eeg=fixed_support) - fixed_clean)[..., :500] ** 2).mean()); initial_diff = float(diff.training_loss(fixed_clean, query_y=fixed_y, support_eeg=fixed_support, generator=generator, timestep=fixed_t, noise=fixed_noise)[0])
    gradient_coverage = {"DET": 0.0, "DIFF": 0.0}
    for phase, model, optimizer, ema in (("DET", det, opt_det, ema_det), ("DIFF", diff, opt_diff, ema_diff)):
        for step, (index, match) in enumerate(schedule):
            query, clean, support = fields(index, match); model.train(); optimizer.zero_grad(set_to_none=True)
            if phase == "DET":
                prediction = det(query_y=query, support_eeg=support); loss = (prediction[..., :500] - clean[..., :500]).square().mean()
            else:
                loss, _ = diff.training_loss(clean, query_y=query, support_eeg=support, generator=generator)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]; active = [p for p in trainable if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0]
            gradient_coverage[phase] = max(gradient_coverage[phase], len(active) / len(trainable)); grad = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0)); optimizer.step(); ema.update(model)
            if step % 200 == 0 or step + 1 == updates: curve.append({"phase": phase, "step": step + 1, "loss": float(loss.detach()), "gradient_norm": grad, "active_gradient_fraction": len(active) / len(trainable)})
    raw_det = {k: v.detach().cpu().clone() for k, v in det.state_dict().items()}; raw_diff = {k: v.detach().cpu().clone() for k, v in diff.state_dict().items()}; ema_det.copy_to(det); ema_diff.copy_to(diff); det.eval(); diff.eval()
    target = _root(c, "model_root") / ("technical" if technical else "models") / str(seed) / f"fold_{fold:02d}"; target.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        cfg, det, diff, ema_det, ema_diff,
        raw_det=raw_det, raw_diff=raw_diff, seed=seed, fold=fold, updates=updates,
        parameter_count_det=pdet, parameter_count_diff=pdiff,
        training_seconds=time.monotonic() - started,
        optimizer_det=opt_det.state_dict(), optimizer_diff=opt_diff.state_dict(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),
        diffusion_generator_state=generator.get_state(),
        numpy_rng_state=rng.bit_generator.state,
    )
    torch.save(payload, target / "checkpoint.pt"); _csv(target / "training_curve.csv", curve)
    with torch.no_grad():
        final_det_pred = det(query_y=fixed_y, support_eeg=fixed_support); final_det_loss = float(((final_det_pred - fixed_clean)[..., :500] ** 2).mean()); final_diff_loss = float(diff.training_loss(fixed_clean, query_y=fixed_y, support_eeg=fixed_support, generator=generator, timestep=fixed_t, noise=fixed_noise)[0]); bank = [torch.randn(fixed_clean.shape, device=device, generator=generator) for _ in range(8)]; diff_samples = [diff.sample(query_y=fixed_y, support_eeg=fixed_support, initial_noise=noise) for noise in bank]; final_diff_pred = torch.stack(diff_samples).mean(0)
    def quality(prediction: Any, truth: Any) -> tuple[float, float]:
        p = prediction[..., :500].detach().cpu().numpy().ravel(); t = truth[..., :500].detach().cpu().numpy().ravel(); return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12)), float(np.corrcoef(p, t)[0, 1])
    det_rr, det_corr = quality(final_det_pred, fixed_clean); diff_rr, diff_corr = quality(final_diff_pred, fixed_clean)
    validation = {"rows": len(validation_index), "diff_beats_raw": None}
    if len(validation_index):
        vy, vx, vs = fields(validation_index, np.ones(len(validation_index), bool)); noises = [torch.randn(vx.shape, device=device, generator=generator) for _ in range(8)]
        with torch.no_grad(): vp = torch.stack([diff.sample(query_y=vy, support_eeg=vs, initial_noise=n) for n in noises]).mean(0)
        validation.update({"raw_rrmse": quality(vy, vx)[0], "diff_rrmse": quality(vp, vx)[0], "diff_beats_raw": quality(vp, vx)[0] < quality(vy, vx)[0]})
    # Exact support-set permutation invariance and deterministic context response.
    perm = torch.arange(15, -1, -1, device=device); wrong_key = next(key for key in contexts.files if key.startswith("p") and not key.startswith(f"p{int(subjects_all[train_index[0]])}_s{int(sessions_all[train_index[0]])}")); wrong_support = torch.as_tensor(np.repeat(contexts[wrong_key][None], len(train_index), axis=0), device=device)
    with torch.no_grad():
        permuted = det(query_y=fixed_y, support_eeg=fixed_support[:, perm]); matched = det(query_y=fixed_y, support_eeg=fixed_support); wrong = det(query_y=fixed_y, support_eeg=wrong_support); shuffled_y = det(query_y=fixed_y.flip(-1), support_eeg=fixed_support)
        replay_a = diff.sample(query_y=fixed_y, support_eeg=fixed_support, initial_noise=bank[0])
        replay_b = diff.sample(query_y=fixed_y, support_eeg=fixed_support, initial_noise=bank[0])
    saved = torch.load(target / "checkpoint.pt", map_location=device, weights_only=False)
    reload_det = DeterministicRawSupportCleaner(cfg).to(device); reload_det.load_state_dict(saved["det"]); reload_det.eval()
    reload_diff = RawSupportCleanDiffusion(cfg).to(device); reload_diff.load_state_dict(saved["diff"]); reload_diff.eval()
    reload_opt_det = AdamW(reload_det.parameters(), lr=float(c["learning_rate"])); reload_opt_det.load_state_dict(saved["optimizer_det"])
    reload_opt_diff = AdamW(reload_diff.parameters(), lr=float(c["learning_rate"])); reload_opt_diff.load_state_dict(saved["optimizer_diff"])
    with torch.no_grad(): reloaded = reload_det(query_y=fixed_y, support_eeg=fixed_support); reloaded_diff = reload_diff.sample(query_y=fixed_y, support_eeg=fixed_support, initial_noise=bank[0])
    query_shuffle_rrmse, _ = quality(shuffled_y, fixed_clean)
    resume_fields = ("optimizer_det", "optimizer_diff", "torch_rng_state", "cuda_rng_state", "diffusion_generator_state", "numpy_rng_state")
    metrics = {"status": "RAW_SUPPORT_MODELS_TRAINED", "technical": technical, "seed": seed, "fold": fold, "updates": updates, "parameter_count_det": pdet, "parameter_count_diff": pdiff, "parameter_difference": pdet - pdiff, "training_seconds": payload["training_seconds"], "gradient_coverage_det": gradient_coverage["DET"], "gradient_coverage_diff": gradient_coverage["DIFF"], "initial_det_loss": initial_det, "final_det_loss": final_det_loss, "det_loss_reduction": 1 - final_det_loss / max(initial_det, 1e-12), "initial_diff_loss": initial_diff, "final_diff_loss": final_diff_loss, "diff_loss_reduction": 1 - final_diff_loss / max(initial_diff, 1e-12), "det_rrmse": det_rr, "det_correlation": det_corr, "diff_k8_rrmse": diff_rr, "diff_k8_correlation": diff_corr, "support_permutation_max_abs": float(torch.max(torch.abs(permuted - matched))), "wrong_context_output_rms_change": float(torch.sqrt(torch.mean((wrong - matched) ** 2))), "query_shuffle_output_rms_change": float(torch.sqrt(torch.mean((shuffled_y - matched) ** 2))), "query_shuffle_rrmse": query_shuffle_rrmse, "query_shuffle_worsens": bool(query_shuffle_rrmse > det_rr), "checkpoint_reload_exact": bool(torch.equal(reloaded, matched) and torch.equal(reloaded_diff, replay_a)), "common_noise_replay_exact": bool(torch.equal(replay_a, replay_b)), "checkpoint_resume_state_complete": bool(all(field in saved for field in resume_fields)), "validation": validation, "evaluator_opened": False}
    _json(target / "metrics.json", metrics); contexts.close(); return metrics


def stage_technical(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    import torch
    result = _train(c, int(c["primary_seed"]), 0, torch.device("cuda"), technical=True)
    passed = result["det_loss_reduction"] >= .95 and result["diff_loss_reduction"] >= .95 and result["det_rrmse"] <= .05 and result["det_correlation"] >= .98 and result["diff_k8_rrmse"] <= .10 and result["diff_k8_correlation"] >= .95 and result["checkpoint_reload_exact"] and result["checkpoint_resume_state_complete"] and result["common_noise_replay_exact"] and result["support_permutation_max_abs"] < 1e-5 and result["wrong_context_output_rms_change"] > 1e-7 and result["query_shuffle_output_rms_change"] > 1e-4 and result["query_shuffle_worsens"] and bool(result["validation"]["diff_beats_raw"]) and min(result["gradient_coverage_det"], result["gradient_coverage_diff"]) >= .99
    result["technical_validity_passed"] = bool(passed); result["status"] = "RAW_SUPPORT_TECHNICAL_PASSED" if passed else "RAW_SUPPORT_TECHNICAL_FAILED"; _json(_root(c, "model_root") / "technical_validity.json", result); _json(run / "result_summary.json", result)
    if not passed: raise RuntimeError(result)
    return result


def _task_seed_fold(c: Mapping[str, Any], task_index: int, *, additional: bool = False) -> tuple[int, int]:
    seeds = list(map(int, c["additional_seeds"])) if additional else [int(c["primary_seed"])]
    return seeds[task_index // 9], task_index % 9


def stage_train(c: Mapping[str, Any], task_index: int, run: Path, *, additional: bool = False) -> dict[str, Any]:
    import torch
    if not json.loads((_root(c, "model_root") / "technical_validity.json").read_text())["technical_validity_passed"]: raise RuntimeError("technical validity gate failed")
    seed, fold = _task_seed_fold(c, task_index, additional=additional); result = _train(c, seed, fold, torch.device("cuda"), technical=False); _json(run / "result_summary.json", result); return result


def _load_clean_model(c: Mapping[str, Any], seed: int, fold: int, device: Any):
    import torch
    from eeg_cgdr.models.raw_support_clean_diffusion import DeterministicRawSupportCleaner, RawSupportCleanConfig, RawSupportCleanDiffusion
    path = _root(c, "model_root") / "models" / str(seed) / f"fold_{fold:02d}" / "checkpoint.pt"; cp = torch.load(path, map_location="cpu", weights_only=False); cfg = RawSupportCleanConfig(**cp["config"]); det = DeterministicRawSupportCleaner(cfg).to(device); diff = RawSupportCleanDiffusion(cfg).to(device); det.load_state_dict(cp["det"]); diff.load_state_dict(cp["diff"]); det.eval(); diff.eval(); return cp, det, diff


def stage_infer(c: Mapping[str, Any], task_index: int, run: Path, *, additional: bool = False) -> dict[str, Any]:
    import torch
    seed, fold = _task_seed_fold(c, task_index, additional=additional); device = torch.device("cuda"); _, det, diff = _load_clean_model(c, seed, fold, device); recipient = fold + 1
    support = np.load(_root(c, "model_root") / "materialized" / f"fold_{fold:02d}" / "support_sets.npz"); base = _seed_fold(_root(c, "strict_root"), int(c["primary_seed"]), fold); manifest = _read(_root(c, "strict_root") / "prepared" / f"fold_{fold:02d}" / "unit_manifest.csv"); runtime = []
    try:
        for unit_index, protocol in enumerate(SAME):
            inf = np.load(base / "units" / protocol / "inference.npz"); target = _root(c, "model_root") / "inference" / str(seed) / f"fold_{fold:02d}" / protocol; target.mkdir(parents=True, exist_ok=True)
            if not int(inf["recipient_eligible"]): np.savez_compressed(target / "inference_outputs.npz"); continue
            session = unit_index + 1; match_key = f"p{recipient}_s{session}"
            if match_key not in support: raise RuntimeError("eligible recipient lacks frozen support set")
            donor_keys = [key for key in support.files if key.startswith("p") and key.endswith(f"_s{session}") and not key.startswith(f"p{recipient}_")]
            contexts = [("POP", np.asarray(support[f"pop_s{session}"])), ("MATCH", np.asarray(support[match_key]))] + [(f"WRONG-{key.split('_')[0][1:]}", np.asarray(support[key])) for key in donor_keys]
            outputs: dict[str, np.ndarray] = {}
            for panel in ("paired", "natural"):
                y = np.asarray(inf[f"{panel}_y"], np.float32); outputs[f"{panel}_RAW"] = y; yt = torch.as_tensor(y, device=device); bank = v11._noise_bank(y.shape, seed + fold * 100000 + unit_index * 10000, 8)
                for name, patches in contexts:
                    st = torch.as_tensor(np.repeat(patches[None], len(y), axis=0), device=device); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
                    with torch.no_grad(): dp = det(query_y=yt, support_eeg=st)
                    torch.cuda.synchronize(); det_seconds = time.monotonic() - started; started = time.monotonic()
                    with torch.no_grad(): samples = [diff.sample(query_y=yt, support_eeg=st, initial_noise=torch.as_tensor(noise, device=device)) for noise in bank]
                    torch.cuda.synchronize(); diff_seconds = time.monotonic() - started; k1 = samples[0].cpu().numpy(); k8 = torch.stack(samples).mean(0).cpu().numpy(); dp = dp.cpu().numpy()
                    for value in (dp, k1, k8): value[..., 500:] = 0
                    outputs[f"{panel}_DET-CLEAN-{name}"] = dp; outputs[f"{panel}_DIFF-CLEAN-{name}-K1"] = k1; outputs[f"{panel}_DIFF-CLEAN-{name}-K8"] = k8
                    runtime.append({"seed": seed, "fold": fold, "participant": recipient, "protocol": protocol, "panel": panel, "context": name, "windows": len(y), "det_seconds": det_seconds, "diff_k8_seconds": diff_seconds, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "det_NFE": 1, "diff_K1_NFE": 25, "diff_K8_NFE": 200})
            np.savez_compressed(target / "inference_outputs.npz", **outputs)
        _csv(_root(c, "model_root") / "inference_manifest" / f"seed_{seed}_fold_{fold:02d}.csv", runtime)
    finally: support.close()
    summary = {"status": "RAW_SUPPORT_EVALUATOR_BLIND_INFERENCE_COMPLETED", "seed": seed, "fold": fold, "evaluator_opened": False}; _json(run / "result_summary.json", summary); return summary


def stage_evaluate(c: Mapping[str, Any], task_index: int, run: Path, *, additional: bool = False) -> dict[str, Any]:
    seed, fold = _task_seed_fold(c, task_index, additional=additional); base = _seed_fold(_root(c, "strict_root"), int(c["primary_seed"]), fold); source = _root(c, "strict_root") / "prepared" / f"fold_{fold:02d}"; outputs = _root(c, "model_root") / "inference" / str(seed) / f"fold_{fold:02d}"; paired, natural = nxt._evaluate_outputs(c, base, source, outputs, seed, fold, "RAW-SUPPORT-CLEAN")
    root = _root(c, "model_root") / "evaluation"; _csv(root / f"seed_{seed}_fold_{fold:02d}_paired.csv", paired); _csv(root / f"seed_{seed}_fold_{fold:02d}_natural.csv", natural); summary = {"status": "RAW_SUPPORT_INDEPENDENT_EVALUATION_COMPLETED", "seed": seed, "fold": fold, "paired_rows": len(paired), "natural_rows": len(natural)}; _json(run / "result_summary.json", summary); return summary


def stage_aggregate(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    root = _root(c, "model_root"); evaluation = root / "evaluation"; paired = []; natural = []
    seeds = [int(c["primary_seed"])] + [seed for seed in map(int, c["additional_seeds"]) if (root / "models" / str(seed)).exists()]
    for seed in seeds:
        for fold in range(9): paired.extend(_read(evaluation / f"seed_{seed}_fold_{fold:02d}_paired.csv")); natural.extend(_read(evaluation / f"seed_{seed}_fold_{fold:02d}_natural.csv"))
    _csv(root / "paired_metrics.csv", paired); _csv(root / "natural_safety.csv", natural); effects = []
    for seed in seeds:
        for participant in range(1, 10):
            take = [r for r in paired if int(r["seed"]) == seed and int(r["participant"]) == participant]; by = defaultdict(list)
            for row in take: by[row["method"]].append(float(row["rrmse"]))
            needed = ("DIFF-CLEAN-POP-K8", "DIFF-CLEAN-MATCH-K8", "DET-CLEAN-POP", "DET-CLEAN-MATCH", "DIFF-CLEAN-MATCH-K1")
            if not all(name in by for name in needed): continue
            wrong = [np.mean(values) for name, values in by.items() if name.startswith("DIFF-CLEAN-WRONG-") and name.endswith("-K8")]
            effects.append({"seed": seed, "participant": participant, "U_P": float(np.mean(by["DIFF-CLEAN-POP-K8"]) - np.mean(by["DIFF-CLEAN-MATCH-K8"])), "U_W": float(np.mean(wrong) - np.mean(by["DIFF-CLEAN-MATCH-K8"])), "E_D": float(np.mean(by["DET-CLEAN-MATCH"]) - np.mean(by["DIFF-CLEAN-MATCH-K8"])), "DeltaSA": float((np.mean(by["DIFF-CLEAN-POP-K8"]) - np.mean(by["DIFF-CLEAN-MATCH-K8"])) - (np.mean(by["DET-CLEAN-POP"]) - np.mean(by["DET-CLEAN-MATCH"]))), "E_K": float(np.mean(by["DIFF-CLEAN-MATCH-K1"]) - np.mean(by["DIFF-CLEAN-MATCH-K8"]))})
    _csv(root / "participant_seed_effects.csv", effects); participant = []
    for p in range(1, 10): participant.append({"participant": p, **{key: float(np.mean([r[key] for r in effects if r["participant"] == p])) for key in ("U_P", "U_W", "E_D", "DeltaSA", "E_K")}})
    _csv(root / "participant_effects.csv", participant); summaries = {key: _summary(c, np.asarray([r[key] for r in participant]), {seed: np.asarray([r[key] for r in effects if r["seed"] == seed]) for seed in seeds}) for key in ("U_P", "U_W", "E_D", "DeltaSA", "E_K")}
    method_rows = []
    for method in sorted({r["method"] for r in paired}):
        per = [np.mean([float(r[key]) for r in paired if r["method"] == method and int(r["participant"]) == p]) for p in range(1, 10) for key in ["rrmse"]]
        take = [r for r in paired if r["method"] == method]; method_rows.append({"method": method, "participants": len({r["participant"] for r in take}), "rrmse": float(np.mean(per)), "correlation": float(np.mean([float(r["correlation"]) for r in take])), "delta_snr": float(np.mean([float(r["delta_snr"]) for r in take])), "paired_spectral_utility": float(np.mean([float(r["paired_spectral_utility"]) for r in take]))})
    _csv(root / "method_summary_participant_first.csv", method_rows)
    safety = []
    for method in sorted({r["method"] for r in natural}):
        for p in range(1, 10):
            take = [r for r in natural if r["method"] == method and int(r["participant"]) == p]
            if take: safety.append({"method": method, "participant": p, **{key: float(np.mean([float(r[key]) for r in take])) for key in ("eog_attenuation", "preservation", "mi_band_distortion", "covariance", "mi_kappa", "erd_preservation")}})
    _csv(root / "participant_natural_safety.csv", safety)
    def natural_mean(method: str): return {key: float(np.mean([r[key] for r in safety if r["method"] == method])) for key in ("eog_attenuation", "preservation", "mi_band_distortion", "covariance", "mi_kappa", "erd_preservation")}
    match = natural_mean("DIFF-CLEAN-MATCH-K8"); pop = natural_mean("DIFF-CLEAN-POP-K8"); margins = {key: match[key] - pop[key] for key in match}
    raw_rr = next(r["rrmse"] for r in method_rows if r["method"] == "RAW"); match_rr = next(r["rrmse"] for r in method_rows if r["method"] == "DIFF-CLEAN-MATCH-K8"); up, uw, ek = summaries["U_P"], summaries["U_W"], summaries["E_K"]
    subject = all(metric["mean"] >= .005 and metric["median"] > 0 and metric["positive"] >= 8 and metric["one_sided_exact_sign_flip"] < .05 for metric in (up, uw)); absolute = match_rr < raw_rr; safe = margins["eog_attenuation"] >= -.01 and margins["preservation"] >= -.02 and margins["covariance"] <= .02 and margins["mi_band_distortion"] <= .02 and margins["mi_kappa"] >= -.02 and margins["erd_preservation"] >= -.02; sampling = ek["mean"] > 0 and ek["median"] > 0 and ek["positive"] >= 7
    det_nat = natural_mean("DET-CLEAN-MATCH"); det_rr = next(r["rrmse"] for r in method_rows if r["method"] == "DET-CLEAN-MATCH"); dominated = match_rr > det_rr and match["preservation"] < det_nat["preservation"] and match["mi_kappa"] < det_nat["mi_kappa"] and match["erd_preservation"] < det_nat["erd_preservation"]
    passed = subject and absolute and safe and sampling and not dominated; one_seed = len(seeds) == 1; route = {"status": "RAW_TEMPORAL_SUPPORT_CLEAN_DIFFUSION_ONE_SEED_PASSED" if passed else ("RAW_TEMPORAL_SUPPORT_CLEAN_DIFFUSION_ROUTE_CLOSED" if not subject else "RAW_TEMPORAL_SUPPORT_CLEAN_DIFFUSION_NOT_ADVANCED"), "additional_seeds_authorized": bool(passed and one_seed), "subject_gate": bool(subject), "absolute_validity": bool(absolute), "natural_safety": bool(safe), "sampling_average_effect": bool(sampling), "pareto_dominated_by_det": bool(dominated), "current_data_unseen_subject_calibration_exploration_ends_if_failed": not passed, "family_wide_negative_forbidden": True, "development_only": True}
    runtime = []
    for path in sorted((root / "inference_manifest").glob("*.csv")): runtime.extend(_read(path))
    compute = [{"method": "DET-CLEAN", "parameters": json.loads((root / "models" / str(seeds[0]) / "fold_00" / "metrics.json").read_text())["parameter_count_det"], "updates": int(c["training_updates"]), "NFE": 1, "mean_seconds": float(np.mean([float(r["det_seconds"]) for r in runtime])), "peak_gpu_memory_mb": float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))}, {"method": "DIFF-CLEAN-K8", "parameters": json.loads((root / "models" / str(seeds[0]) / "fold_00" / "metrics.json").read_text())["parameter_count_diff"], "updates": int(c["training_updates"]), "NFE": 200, "mean_seconds": float(np.mean([float(r["diff_k8_seconds"]) for r in runtime])), "peak_gpu_memory_mb": float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))}]
    _csv(root / "method_compute.csv", compute); summary = {"effects": summaries, "method_summary": method_rows, "natural_match": match, "natural_pop": pop, "natural_margins": margins, "routing": route, "availability": {"eligible_units": 26, "denominator": 27, "participants": 9}, "scientific_unit": "participant", "seeds": seeds}; _json(root / "result_summary.json", summary); _json(root / "route_decision.json", route)
    figdir = root / "figures"; figdir.mkdir(parents=True, exist_ok=True); fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(1, 10); ax.axhline(0, color="black", lw=.8); ax.plot(x, [r["U_P"] for r in participant], "o-", label="U_P"); ax.plot(x, [r["U_W"] for r in participant], "s-", label="U_W"); ax.set(xlabel="Participant", ylabel="Positive RRMSE utility"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "subject_effects.png", dpi=180); plt.close(fig)
    _json(run / "result_summary.json", summary); return summary


def stage_final(c: Mapping[str, Any], task_index: int, run: Path) -> dict[str, Any]:
    geometry = json.loads((_root(c, "geometry_root") / "corrected_geometry_summary.json").read_text()); oracle = json.loads((_root(c, "geometry_root") / "oracle_actionability_summary.json").read_text()); model_path = _root(c, "model_root") / "result_summary.json"; model = json.loads(model_path.read_text()) if model_path.exists() else None
    route = {"eog_transfer_personalization_family": "closed", "covariance_actionability": oracle["routing"]["status"], "raw_temporal_support_clean_diffusion": model["routing"]["status"] if model else "NOT_RUN_TECHNICAL_FAILURE", "development_only": True, "family_wide_negative_forbidden": True}; summary = {"corrected_geometry": geometry, "oracle_actionability": oracle, "raw_support_clean_diffusion": model, "routing": route}; _json(_root(c, "result_root") / "result_summary.json", summary); _json(_root(c, "result_root") / "route_decision.json", route)
    lines = ["# Corrected neural geometry and oracle actionability", "", "The historical covariance label is corrected to **robust spatial-correlation geometry** because each channel is median/MAD standardized. Support and later-query evaluator windows both use the frozen lowest-30% EOG-energy rule. AIRM populations use a true affine-invariant Karcher barycenter; log-Euclidean populations use a log-domain barycenter. Historical outputs remain unchanged.", "", f"Oracle route: `{oracle['routing']['status']}`. The query-clean covariance context is evaluator-only and is not deployable.", ""]
    Path("reports/corrected_neural_geometry_oracle.md").write_text("\n".join(lines), encoding="utf-8")
    if model:
        effects = model["effects"]; report = ["# Raw temporal support-conditioned clean EEG diffusion", "", f"Decision: `{model['routing']['status']}` on BCI2b same-session development data.", "", "The diffusion state is the full clean EEG waveform. Query inference sees corrupted EEG and sixteen raw temporal EEG support patches; it never sees query EOG, labels, participant ID, covariance summaries, or clean targets.", ""]
        report.extend(f"- {key}: mean {value['mean']:+.5f}, median {value['median']:+.5f}, {value['positive']}/9, one-sided exact p={value['one_sided_exact_sign_flip']:.6f}." for key, value in effects.items()); report.extend(["", f"Natural MATCH means: EOG attenuation {model['natural_match']['eog_attenuation']:+.5f}, preservation {model['natural_match']['preservation']:.5f}, covariance {model['natural_match']['covariance']:.5f}, MI-band distortion {model['natural_match']['mi_band_distortion']:.5f}, MI kappa {model['natural_match']['mi_kappa']:.5f}.", "", "This is development evidence only. A failure constrains this frozen raw-support encoder/cross-attention clean-conditional instance, not diffusion or personalization families."])
        Path("reports/raw_support_clean_diffusion.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _json(run / "result_summary.json", route); return route


def stage_train_primary(c, i, r): return stage_train(c, i, r, additional=False)
def stage_train_additional(c, i, r): return stage_train(c, i, r, additional=True)
def stage_infer_primary(c, i, r): return stage_infer(c, i, r, additional=False)
def stage_infer_additional(c, i, r): return stage_infer(c, i, r, additional=True)
def stage_evaluate_primary(c, i, r): return stage_evaluate(c, i, r, additional=False)
def stage_evaluate_additional(c, i, r): return stage_evaluate(c, i, r, additional=True)


def run_stage(config_path: Path, stage: str, run_dir: Path, *, task_index: int = 0) -> dict[str, Any]:
    c = _config(config_path); run_dir.mkdir(parents=True, exist_ok=True)
    stages = {"audit": stage_audit, "geometry-support": stage_geometry_support, "geometry-evaluate": stage_geometry_evaluate, "geometry-aggregate": stage_geometry_aggregate, "oracle-action": stage_oracle_action, "oracle-aggregate": stage_oracle_aggregate, "materialize": stage_materialize, "technical": stage_technical, "train": stage_train_primary, "train-additional": stage_train_additional, "infer": stage_infer_primary, "infer-additional": stage_infer_additional, "evaluate": stage_evaluate_primary, "evaluate-additional": stage_evaluate_additional, "aggregate": stage_aggregate, "final": stage_final}
    if stage not in stages: raise ValueError(stage)
    return stages[stage](c, task_index, run_dir)


__all__ = ["_karcher_mean", "_log_barycenter", "_robust_correlation_geometry", "_sign_flip", "run_stage"]
