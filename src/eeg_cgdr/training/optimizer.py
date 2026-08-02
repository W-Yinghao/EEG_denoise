"""Truthful mixed-precision optimizer-step accounting."""

from __future__ import annotations

from typing import Any


def scaler_optimizer_step_succeeded(scaler: Any, optimizer: Any) -> bool:
    """Run one scaler step and report whether the optimizer actually advanced.

    ``GradScaler.step`` silently skips ``optimizer.step`` when it finds
    non-finite gradients.  A skipped step reduces the scale; a successful step
    leaves it unchanged or grows it.  Counting only successful calls keeps the
    advertised optimizer-update budget truthful.
    """

    previous_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= previous_scale
