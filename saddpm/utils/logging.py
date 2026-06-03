"""Run logging to a local CSV and (optionally) Weights & Biases (handoff §6).

CSV logging is always on (offline-safe). W&B is opt-in and degrades gracefully if the package or
network is unavailable, so sanity runs never depend on it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """Logs scalar metrics to a CSV file and, optionally, to W&B."""

    def __init__(
        self,
        run_dir: str | Path,
        csv_name: str = "metrics.csv",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: str = "online",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / csv_name
        self._fieldnames: Optional[List[str]] = None
        self._csv_file = None
        self._csv_writer = None

        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    mode=wandb_mode,
                    config=config or {},
                    dir=str(self.run_dir),
                )
            except Exception as exc:  # noqa: BLE001 - never let logging crash training
                print(f"[RunLogger] W&B disabled ({type(exc).__name__}: {exc}); CSV only.")
                self._wandb = None

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """Append one row of metrics at ``step`` to the CSV and W&B."""
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        self._csv_writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._csv_file.flush()
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def log_image(self, name: str, path: str | Path, step: Optional[int] = None) -> None:
        """Log an image file to W&B (no-op if W&B is off)."""
        if self._wandb is not None:
            import wandb

            self._wandb.log({name: wandb.Image(str(path))}, step=step)

    def finish(self) -> None:
        """Close the CSV file and the W&B run."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None
