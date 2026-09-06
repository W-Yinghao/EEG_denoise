"""V41R two-regressor artifact-transfer registry and dynamic paired episodes.

The stored source files contain four eye-region electrodes.  V41R deliberately
reduces them to the registered bipolar VEOG/HEOG coordinates before fitting any
transfer.  Support estimates use only a duration prefix; independent Qgen fits
are evaluator/generator assets and never model inputs.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_signal, fold_eeg_scale
from eeg_scad.data.v24_coordinate_contract import robust_center_scale


EYE_NAMES = ("HEOGL", "HEOGR", "VEOGU", "VEOGL")
BIPOLAR_NAMES = ("VEOG", "HEOG")


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def bipolar_eog(eog: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Return exactly ``[VEOGU-VEOGL, HEOGL-HEOGR]``."""
    labels = [str(value) for value in names]
    if set(labels) != set(EYE_NAMES) or len(labels) != 4:
        raise ValueError(f"expected the four registered eye electrodes, got {labels}")
    index = {name: labels.index(name) for name in EYE_NAMES}
    value = np.asarray(eog, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != 4:
        raise ValueError("eye-electrode data must have shape 4 x time")
    return np.stack((value[index["VEOGU"]] - value[index["VEOGL"]],
                     value[index["HEOGL"]] - value[index["HEOGR"]]))


def ridge_transfer(eeg: np.ndarray, eog: np.ndarray, ratio: float = 0.05) -> tuple[np.ndarray, dict[str, float]]:
    """Fit ``Y E' (E E' + lambda I)^-1`` after temporal centering."""
    y = np.asarray(eeg, dtype=np.float64)
    e = np.asarray(eog, dtype=np.float64)
    if y.ndim != 2 or e.ndim != 2 or e.shape[0] != 2 or y.shape[1] != e.shape[1]:
        raise ValueError("ridge transfer requires CxT EEG and exactly 2xT EOG")
    yc = y - y.mean(axis=1, keepdims=True)
    ec = e - e.mean(axis=1, keepdims=True)
    gram = ec @ ec.T
    ridge = float(ratio) * max(float(np.trace(gram) / 2), np.finfo(float).eps)
    transfer = (yc @ ec.T) @ np.linalg.inv(gram + ridge * np.eye(2))
    fitted = transfer @ ec
    residual = yc - fitted
    r2 = 1 - float(np.sum(residual * residual) / max(np.sum(yc * yc), np.finfo(float).eps))
    return transfer, {
        "ridge": ridge,
        "fit_r2": r2,
        "condition_number": float(np.linalg.cond(gram + ridge * np.eye(2))),
    }


@dataclass(frozen=True)
class TransferCell:
    owner: str
    session: str
    task: str
    support_seconds: int
    transfer: np.ndarray
    query_transfer: np.ndarray
    eog_center: np.ndarray
    eog_scale: np.ndarray
    quality: np.ndarray
    starts: tuple[int, ...]
    support_digest: str
    query_digest: str


class TransferRegistry:
    """Fold-local registry with training-only signature normalization."""

    def __init__(self, data: Mapping[str, Any], fold: Mapping[str, Any], seconds: int = 30,
                 ridge_ratio: float = 0.05, include_query_transfer: bool = True) -> None:
        if seconds not in (10, 30):
            raise ValueError("nonzero registered support duration must be 10 or 30 seconds")
        self.data, self.fold, self.seconds, self.ridge_ratio = data, fold, seconds, ridge_ratio
        self.include_query_transfer = include_query_transfer
        self.root = Path(data["v19_derived_root"])
        self.eeg_scale = fold_eeg_scale(data, fold["train"])
        self.cells: dict[tuple[str, str, str], TransferCell] = {}
        owners = sorted(set(fold["train"] + fold["validation"] + fold["test"] + [str(data["auxiliary_support_owner"])]))
        for owner, session, task in itertools.product(owners, data["sessions"], data["tasks"]):
            try:
                self.cells[(owner, session, task)] = self._fit_cell(owner, session, task)
            except FileNotFoundError:
                continue
        train_cells = [cell for key, cell in self.cells.items() if key[0] in fold["train"]]
        if not train_cells:
            raise RuntimeError("no outer-training transfer cells")
        raw = np.concatenate([self._continuous(cell.transfer, cell.quality) for cell in train_cells], axis=0)
        self.continuous_center = raw.mean(axis=0)
        self.continuous_scale = raw.std(axis=0).clip(1e-6)
        self.population_transfer: dict[tuple[str, str], np.ndarray] = {}
        self.population_quality: dict[tuple[str, str], np.ndarray] = {}
        for session, task in itertools.product(data["sessions"], data["tasks"]):
            block = [cell for key, cell in self.cells.items() if key[0] in fold["train"] and key[1:] == (session, task)]
            if block:
                self.population_transfer[(session, task)] = np.mean([cell.transfer for cell in block], axis=0)
                self.population_quality[(session, task)] = np.mean([cell.quality for cell in block], axis=0)

    def _load(self, owner: str, session: str, task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        path = self.root / "prepared" / owner / f"{session}_{task}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            return (np.asarray(archive["eeg"], np.float64), np.asarray(archive["eog"], np.float64),
                    [str(value) for value in archive["eog_names"]])

    def _fit_cell(self, owner: str, session: str, task: str) -> TransferCell:
        eeg, eye, names = self._load(owner, session, task)
        eog = bipolar_eog(eye, names)
        prefix = self.seconds * int(self.data.get("sampling_rate", 100))
        if min(eeg.shape[1], eog.shape[1]) < int(self.data["qgen_stop"]):
            raise FileNotFoundError("record shorter than Qgen")
        center, scale = robust_center_scale(eog[:, :prefix])
        support_eog = (eog[:, :prefix] - center[:, None]) / scale[:, None]
        support_eeg = eeg[:, :prefix] / self.eeg_scale[:, None]
        transfer, diagnostics = ridge_transfer(support_eeg, support_eog, self.ridge_ratio)
        q0, q1 = int(self.data["qgen_start"]), int(self.data["qgen_stop"])
        if self.include_query_transfer:
            query_eog = (eog[:, q0:q1] - center[:, None]) / scale[:, None]
            query_eeg = eeg[:, q0:q1] / self.eeg_scale[:, None]
            query_transfer, _ = ridge_transfer(query_eeg, query_eog, self.ridge_ratio)
            query_digest = array_sha256(np.concatenate((query_eeg, query_eog)))
        else:
            query_transfer = np.full_like(transfer, np.nan)
            query_digest = "NOT_READ"
        rms = np.sqrt(np.mean((eog[:, :prefix] - center[:, None]) ** 2, axis=1)).clip(1e-8)
        quality = np.array([np.log(rms[0]), np.log(rms[1]), diagnostics["fit_r2"],
                            np.log1p(diagnostics["condition_number"])], dtype=np.float64)
        starts = tuple(range(0, prefix - int(self.data["support_window_samples"]) + 1,
                             int(self.data["support_window_samples"])))
        return TransferCell(owner, session, task, self.seconds, transfer, query_transfer, center, scale,
                            quality, starts, array_sha256(np.concatenate((support_eeg, support_eog))),
                            query_digest)

    @staticmethod
    def _continuous(transfer: np.ndarray, quality: np.ndarray) -> np.ndarray:
        norm = np.log(np.linalg.norm(transfer, axis=1, keepdims=True).clip(1e-8))
        return np.concatenate((transfer, norm, np.broadcast_to(quality[None], (len(transfer), 4))), axis=1)

    def signature(self, owner: str, session: str, task: str, condition: str = "MATCH") -> np.ndarray:
        key = (owner, session, task)
        if condition == "POP":
            transfer = self.population_transfer[(session, task)]
            quality = self.population_quality[(session, task)]
        elif condition == "ORACLE":
            cell = self.cells[key]
            if not self.include_query_transfer:
                raise RuntimeError("ORACLE unavailable in inference-only registry")
            transfer, quality = cell.query_transfer, cell.quality
        else:
            cell = self.cells[key]
            transfer, quality = cell.transfer, cell.quality
        continuous = (self._continuous(transfer, quality) - self.continuous_center) / self.continuous_scale
        identity = np.eye(len(transfer), dtype=np.float64)
        return np.concatenate((continuous, identity), axis=1).astype(np.float32)

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows = []
        for key, cell in sorted(self.cells.items()):
            role = next(role for role in ("train", "validation", "test") if cell.owner in self.fold[role]) if cell.owner != str(self.data["auxiliary_support_owner"]) else "auxiliary"
            rows.append({
                "fold": self.fold["fold"], "participant": cell.owner, "role": role,
                "session": cell.session, "task": cell.task, "support_seconds": cell.support_seconds,
                "window_count": len(cell.starts), "starts": ";".join(map(str, cell.starts)),
                "overlap_samples": 0, "repeated_samples": 0,
                "normalization_samples": cell.support_seconds * int(self.data.get("sampling_rate", 100)),
                "eog_regressors": 2, "eog_order": "VEOG;HEOG",
                "veog_formula": "VEOGU-VEOGL", "heog_formula": "HEOGL-HEOGR",
                "transfer_shape": "46x2", "support_digest": cell.support_digest,
                "query_transfer_digest": cell.query_digest, "query_transfer_interval": "15000:27000",
                "query_transfer_in_support_estimate": 0, "query_eog_inference_reads": 0,
            })
        return rows


class TransferEpisodeSampler:
    """Dynamic clean/EOG/operator pairing with independent Qgen transfer."""

    def __init__(self, data: Mapping[str, Any], fold: Mapping[str, Any], split: str, seed: int,
                 registry: TransferRegistry) -> None:
        self.data, self.fold, self.split, self.registry = data, fold, split, registry
        self.rng = np.random.Generator(np.random.PCG64DXSM(seed + int(fold["fold"]) * 1000 + {"train": 0, "validation": 1, "test": 2}[split]))
        self.recipients = [key for key in registry.cells if key[0] in fold[split] and key[1:] in registry.population_transfer]
        self.sources = []
        for owner, session, task in itertools.product(fold["train"], data["sessions"], data["tasks"]):
            try:
                eeg, eye, names = registry._load(owner, session, task)
            except FileNotFoundError:
                continue
            if eeg.shape[1] >= int(data["qnatural_start"]) + int(data["window_samples"]):
                self.sources.append((owner, session, task, eeg, bipolar_eog(eye, names)))
        if not self.recipients or len(self.sources) < 2:
            raise RuntimeError(f"incomplete transfer stream fold {fold['fold']} split {split}")
        owners = sorted({key[0] for key in registry.cells})
        perm = np.asarray(owners)[np.random.default_rng(20261141 + int(fold["fold"])).permutation(len(owners))]
        self.shuffled_owner = dict(zip(owners, perm))

    def _source(self, excluded: set[str], session: str, task: str) -> tuple[str, np.ndarray, np.ndarray]:
        choices = [row for row in self.sources if row[0] not in excluded and row[1] == session and row[2] == task]
        if not choices:
            choices = [row for row in self.sources if row[0] not in excluded]
        row = choices[int(self.rng.integers(len(choices)))]
        return row[0], row[3], row[4]

    def _window(self, value: np.ndarray) -> np.ndarray:
        length, low = int(self.data["window_samples"]), int(self.data["qnatural_start"])
        start = int(self.rng.integers(low, max(low + 1, value.shape[1] - length + 1)))
        return np.asarray(value[:, start:start + length], dtype=np.float64)

    def condition_signature(self, meta: Mapping[str, Any], condition: str) -> tuple[np.ndarray, str]:
        owner, session, task = str(meta["participant"]), str(meta["session"]), str(meta["task"])
        if condition in ("POP", "CHANNEL_ONLY"):
            signature = self.registry.signature(owner, session, task, "POP")
            if condition == "CHANNEL_ONLY":
                signature[:, :7] = 0
            return signature, "POP"
        if condition in ("MATCH", "ORACLE"):
            return self.registry.signature(owner, session, task, condition), owner
        eligible = sorted(candidate for candidate in {key[0] for key in self.registry.cells}
                          if candidate != owner and (candidate, session, task) in self.registry.cells)
        if condition == "WRONG":
            other = eligible[0]
        elif condition == "SHUFFLED":
            other = self.shuffled_owner.get(owner, owner)
            if other == owner or (other, session, task) not in self.registry.cells:
                other = eligible[-1]
        else:
            raise ValueError(condition)
        return self.registry.signature(other, session, task, "MATCH"), other

    def sample(self, count: int, zero_proportion: float = 0.15,
               recipient_keys: Sequence[tuple[str, str, str]] | None = None) -> dict[str, Any]:
        arrays = {key: [] for key in ("x", "y", "artifact", "signature")}
        meta = []
        for index in range(count):
            if recipient_keys is None:
                recipient, session, task = self.recipients[int(self.rng.integers(len(self.recipients)))]
            else:
                recipient, session, task = recipient_keys[index % len(recipient_keys)]
            cell = self.registry.cells[(recipient, session, task)]
            clean_owner, clean_eeg, _ = self._source({recipient}, session, task)
            eog_owner, _, source_eog = self._source({recipient, clean_owner}, session, task)
            x = self._window(clean_eeg) / self.registry.eeg_scale[:, None]
            physical_eog = self._window(source_eog)
            latent = (physical_eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
            gain = float(self.rng.choice((0.35, 0.70, 1.15)) * self.rng.uniform(0.85, 1.15))
            artifact = cell.query_transfer @ (gain * latent)
            zero = bool(self.rng.random() < zero_proportion)
            if zero:
                artifact = np.zeros_like(artifact)
            y = x + artifact
            signature = self.registry.signature(recipient, session, task, "MATCH")
            for key, value in (("x", x), ("y", y), ("artifact", artifact), ("signature", signature)):
                arrays[key].append(np.asarray(value, np.float32))
            meta.append({
                "participant": recipient, "session": session, "task": task,
                "clean_owner": clean_owner, "eog_owner": eog_owner,
                "operator_recipient": recipient, "wrong_owner": self.condition_signature({"participant": recipient, "session": session, "task": task}, "WRONG")[1],
                "strict_three_way": int(len({recipient, clean_owner, eog_owner}) == 3),
                "gain": gain, "zero_artifact": int(zero),
                "query_transfer_source": "independent_Qgen", "query_transfer_in_model_condition": 0,
            })
        return {**{key: np.stack(value) for key, value in arrays.items()}, "meta": meta}

    def sample_balanced(self, per_participant: int, zero_proportion: float = 0.15) -> dict[str, Any]:
        participants = sorted(set(key[0] for key in self.recipients))
        keys = []
        for participant in participants:
            cells = sorted(key for key in self.recipients if key[0] == participant)
            keys.extend(cells[index % len(cells)] for index in range(per_participant))
        return self.sample(len(keys), zero_proportion=zero_proportion, recipient_keys=keys)


def flatten_channels(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("multichannel batch must have shape B,C,T")
    return array.reshape(array.shape[0] * array.shape[1], 1, array.shape[2])


def flatten_signatures(signature: np.ndarray) -> np.ndarray:
    array = np.asarray(signature)
    if array.ndim != 3 or array.shape[1:] != (46, 53):
        raise ValueError("signature batch must have shape B,46,53")
    return array.reshape(array.shape[0] * 46, 53)


def reassemble_channels(value: np.ndarray, episodes: int, channels: int = 46) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] != episodes * channels or array.shape[1] != 1:
        raise ValueError("flat output does not match episode/channel contract")
    return array.reshape(episodes, channels, array.shape[-1])


__all__ = [
    "BIPOLAR_NAMES", "EYE_NAMES", "TransferCell", "TransferEpisodeSampler", "TransferRegistry",
    "array_sha256", "bipolar_eog", "flatten_channels", "flatten_signatures", "reassemble_channels",
    "ridge_transfer",
]
