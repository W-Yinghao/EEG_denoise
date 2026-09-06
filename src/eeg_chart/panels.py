"""Paired panels and per-cell transport assets for the three ceiling panels.

Every panel yields the same structure so U1-a/U1-b iterate uniformly:
  cells: list of PanelCell (one per subject/record/session-task cell) with
         support/query operators, lifted covariances, and paired episodes.
Sealed cohorts are never touched (MobileBCI dev-16 only; Klados/BCI2b/SGEYESUB
have no sealed subsets registered; SHU sessions 1-3 only; PhysioMotion dev only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eeg_scad.data.artifact_transfer_v41r import bipolar_eog, ridge_transfer
from eeg_scad.data.v24_coordinate_contract import robust_center_scale
from eeg_chart.positions import bci2b_positions, klados_positions, mobilebci_positions
from eeg_chart.transport import ledoit_wolf_covariance, sh_lift


WINDOW = 512


@dataclass
class PanelCell:
    panel: str
    subject: str
    cell: str
    a_support: np.ndarray                 # C x r support-fit operator
    a_halves: tuple[np.ndarray, np.ndarray]
    a_query: np.ndarray                   # C x r generator/oracle operator
    sigma_support: np.ndarray             # K x K lifted low-EOG covariance (support)
    sigma_query: np.ndarray               # K x K (query region; evaluator-only)
    episodes: list[dict[str, Any]] = field(default_factory=list)


def _low_eog_mask(drive: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    energy = np.sqrt(np.mean(drive * drive, axis=0))
    return energy <= np.quantile(energy, quantile)


def _lifted_cov(lift: np.ndarray, eeg_scaled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kept = eeg_scaled[:, mask] if mask.sum() >= 64 else eeg_scaled
    return ledoit_wolf_covariance(lift @ kept)


def _episode_rows(x: np.ndarray, y: np.ndarray, drive: np.ndarray) -> dict[str, Any]:
    artifact = y - x
    severity = float(np.linalg.norm(artifact) / max(np.linalg.norm(x), 1e-12))
    eog_energy = float(np.sqrt(np.mean(drive * drive)))
    return {"x": x, "y": y, "drive": drive, "severity": severity, "eog_energy": eog_energy}


# ------------------------------------------------------------------ MobileBCI

def build_mobilebci_panel(seeds=(20261201, 20261202)) -> tuple[list[PanelCell], np.ndarray]:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    lift = sh_lift(mobilebci_positions())
    cells: dict[tuple[str, str, str], PanelCell] = {}
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        for participant in fold["test"]:
            for key in sorted(registry30.cells):
                if key[0] != participant:
                    continue
                eeg, eye, names = registry30._load(*key)
                eog = bipolar_eog(eye, names)
                cell30 = registry30.cells[key]
                scaled = eeg / registry30.eeg_scale[:, None]
                support = slice(0, 12000)
                query = slice(int(data["qgen_start"]), int(data["qgen_stop"]))
                center, scale = robust_center_scale(eog[:, support])
                latent_support = (eog[:, support] - center[:, None]) / scale[:, None]
                latent_query = (eog[:, query] - cell30.eog_center[:, None]) / cell30.eog_scale[:, None]
                halves = tuple(
                    ridge_transfer(scaled[:, part], latent_support[:, part_rel], .05)[0]
                    for part, part_rel in ((slice(0, 6000), slice(0, 6000)),
                                           (slice(6000, 12000), slice(6000, 12000))))
                cells[key] = PanelCell(
                    panel="mobilebci", subject=participant, cell="|".join(key),
                    a_support=eb120.cells[key].transfer, a_halves=halves,
                    a_query=cell30.query_transfer,
                    sigma_support=_lifted_cov(lift, scaled[:, support],
                                              _low_eog_mask(latent_support)),
                    sigma_query=_lifted_cov(lift, scaled[:, query],
                                            _low_eog_mask(latent_query)))
        for seed in seeds:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            for x, y, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
                key = (meta["participant"], meta["session"], meta["task"])
                drive = np.linalg.pinv(registry30.cells[key].query_transfer) @ np.asarray(artifact, np.float64)
                cells[key].episodes.append(_episode_rows(np.asarray(x, np.float64),
                                                         np.asarray(y, np.float64), drive))
    return list(cells.values()), lift


# -------------------------------------------------------------------- Klados

def build_klados_panel() -> tuple[list[PanelCell], np.ndarray]:
    import yaml
    from eeg_cgdr.data.klados import load_klados_records

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/cgdr/klados_v4.yaml").read_text())
    records = load_klados_records(config)
    lift = sh_lift(klados_positions())
    cells = []
    for record in records:
        clean = np.asarray(record.clean, np.float64)
        contaminated = np.asarray(record.contaminated, np.float64)
        eog = np.stack((np.asarray(record.veog, np.float64).reshape(-1),
                        np.asarray(record.heog, np.float64).reshape(-1)))
        length = min(clean.shape[1], contaminated.shape[1], eog.shape[1])
        clean, contaminated, eog = clean[:, :length], contaminated[:, :length], eog[:, :length]
        half = length // 2
        _, eeg_scale = robust_center_scale(contaminated[:, :half])
        clean_s, cont_s = clean / eeg_scale[:, None], contaminated / eeg_scale[:, None]
        center, scale = robust_center_scale(eog[:, :half])
        latent = (eog - center[:, None]) / scale[:, None]
        quarter = half // 2
        a_support = ridge_transfer(cont_s[:, :half], latent[:, :half], .05)[0]
        halves = (ridge_transfer(cont_s[:, :quarter], latent[:, :quarter], .05)[0],
                  ridge_transfer(cont_s[:, quarter:half], latent[:, quarter:half], .05)[0])
        artifact_s = cont_s - clean_s
        a_query = ridge_transfer(artifact_s[:, half:], latent[:, half:], .05)[0]
        episodes = []
        for start in range(half, length - WINDOW + 1, WINDOW):
            episodes.append(_episode_rows(clean_s[:, start:start + WINDOW],
                                          cont_s[:, start:start + WINDOW],
                                          latent[:, start:start + WINDOW]))
        cells.append(PanelCell(
            panel="klados", subject=f"P{(record.record_id - 1) % 27 + 1:02d}",
            cell=f"rec{record.record_id:02d}",
            a_support=a_support, a_halves=halves, a_query=a_query,
            sigma_support=_lifted_cov(lift, cont_s[:, :half], _low_eog_mask(latent[:, :half])),
            sigma_query=_lifted_cov(lift, cont_s[:, half:], _low_eog_mask(latent[:, half:])),
            episodes=episodes))
    return cells, lift


# --------------------------------------------------------------------- BCI2b

BCI2B_ROOT = Path("/projects/EEG-foundation-model/BCI-IV")


def _load_bci2b_session(path: Path):
    import mne

    raw = mne.io.read_raw_gdf(path, preload=True, verbose="error")
    picks = np.asarray(raw.get_data(), np.float64) * 1e6
    eeg, eog = picks[:3], picks[3:6]
    eeg[np.isnan(eeg)] = 0.0
    eog[np.isnan(eog)] = 0.0
    try:
        events, _ = mne.events_from_annotations(raw, verbose="error")
        trial_starts = [int(sample) for sample, _, code in events]
    except Exception:
        trial_starts = []
    first_trial = min(trial_starts) if trial_starts else eeg.shape[1] // 3
    sfreq = float(raw.info["sfreq"])
    support_stop = max(int(first_trial - 5 * sfreq), int(30 * sfreq))
    return eeg, eog, support_stop


def build_bci2b_panel() -> tuple[list[PanelCell], np.ndarray]:
    lift = sh_lift(bci2b_positions())
    cells = []
    rng = np.random.default_rng(20260814)
    for subject in range(1, 10):
        sessions = {}
        for run, tag in ((1, "T"), (2, "T"), (3, "T")):
            matches = sorted(BCI2B_ROOT.glob(f"B{subject:02d}{run:02d}{tag}.gdf"))
            if matches:
                sessions[run] = _load_bci2b_session(matches[0])
        if len(sessions) < 3:
            continue
        support_eeg = np.concatenate([sessions[run][0][:, :sessions[run][2]] for run in (1, 2)], axis=1)
        support_eog = np.concatenate([sessions[run][1][:, :sessions[run][2]] for run in (1, 2)], axis=1)
        _, eeg_scale = robust_center_scale(support_eeg)
        center, scale = robust_center_scale(support_eog)
        # The 3 monopolar EOG channels are reduced to the support-fit top-2
        # principal components so every panel shares the r=2 ocular frame.
        standardized_support = (support_eog - center[:, None]) / scale[:, None]
        pca = np.linalg.svd(np.cov(standardized_support), full_matrices=False)[0][:, :2].T  # 2 x 3
        standardize = lambda value: pca @ ((value - center[:, None]) / scale[:, None])
        support_scaled = support_eeg / eeg_scale[:, None]
        latent_support = standardize(support_eog)
        a_support = ridge_transfer(support_scaled, latent_support, .05)[0]
        halves = tuple(ridge_transfer(sessions[run][0][:, :sessions[run][2]] / eeg_scale[:, None],
                                      standardize(sessions[run][1][:, :sessions[run][2]]), .05)[0]
                       for run in (1, 2))
        gen_eeg, gen_eog, _ = sessions[3]
        a_query = ridge_transfer(gen_eeg / eeg_scale[:, None], standardize(gen_eog), .05)[0]
        episodes = []
        for run in (1, 2):
            eeg, eog, support_stop = sessions[run]
            query_eeg = eeg[:, support_stop:] / eeg_scale[:, None]
            query_latent = standardize(eog[:, support_stop:])
            energy = np.sqrt(np.mean(query_latent * query_latent, axis=0))
            starts = np.arange(0, query_eeg.shape[1] - WINDOW, WINDOW)
            if len(starts) < 4:
                continue
            window_energy = np.asarray([energy[s:s + WINDOW].mean() for s in starts])
            low_windows = starts[window_energy <= np.quantile(window_energy, .3)]
            high_windows = starts[window_energy >= np.quantile(window_energy, .7)]
            if len(low_windows) == 0 or len(high_windows) == 0:
                continue
            for index, start in enumerate(low_windows[:12]):
                x = query_eeg[:, start:start + WINDOW]
                donor = high_windows[int(rng.integers(len(high_windows)))]
                drive = query_latent[:, donor:donor + WINDOW]
                y = x + a_query @ drive
                episodes.append(_episode_rows(x, y, drive))
        cells.append(PanelCell(
            panel="bci2b", subject=f"B{subject:02d}", cell=f"B{subject:02d}",
            a_support=a_support, a_halves=halves, a_query=a_query,
            sigma_support=_lifted_cov(lift, support_scaled, _low_eog_mask(latent_support)),
            sigma_query=_lifted_cov(lift, gen_eeg / eeg_scale[:, None],
                                    _low_eog_mask(standardize(gen_eog))),
            episodes=episodes))
    return cells, lift


__all__ = ["PanelCell", "build_bci2b_panel", "build_klados_panel", "build_mobilebci_panel"]
