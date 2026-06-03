"""Typed configuration for BCI-IV-2a data loading and preprocessing.

Values are loaded from ``configs/data.yaml`` so that no preprocessing constant is hard-coded
in the pipeline code (handoff working-rule: all hyperparameters live in YAML/dataclass configs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Static facts about the dataset (paper §3)."""

    name: str = "BNCI2014_001"
    n_subjects: int = 9
    n_classes: int = 4
    sfreq: int = 250
    n_eeg_channels: int = 22
    n_eog_channels: int = 3


@dataclass(frozen=True)
class PreprocessConfig:
    """Filtering / resampling parameters (paper §3.2–3.4)."""

    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 50.0
    fir_design: str = "firwin"
    filter_method: str = "fir"
    notch_hz: float = 50.0
    resample_hz: int = 250


@dataclass(frozen=True)
class EpochConfig:
    """Trial epoching window relative to the class-cue event (motor-imagery interval)."""

    tmin_s: float = 2.0
    tmax_s: float = 6.0


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window + normalisation parameters (paper §3.5–3.6, [DD-6])."""

    length_s: float = 2.0
    step_s: float = 0.5
    pad_to: int = 512
    zscore_per_channel: bool = True

    def length_samples(self, sfreq: int) -> int:
        """Window length in samples for the given sampling rate (e.g. 500 @ 250 Hz)."""
        return int(round(self.length_s * sfreq))

    def step_samples(self, sfreq: int) -> int:
        """Window step in samples for the given sampling rate (e.g. 125 @ 250 Hz)."""
        return int(round(self.step_s * sfreq))


@dataclass(frozen=True)
class DataConfig:
    """Top-level data config aggregating all sub-configs."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    seed: int = 42
    mne_data_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """Load a :class:`DataConfig` from a YAML file (see ``configs/data.yaml``)."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            preprocess=PreprocessConfig(**raw.get("preprocess", {})),
            epoch=EpochConfig(**raw.get("epoch", {})),
            window=WindowConfig(**raw.get("window", {})),
            seed=raw.get("seed", 42),
            mne_data_path=raw.get("mne_data_path", None),
        )
