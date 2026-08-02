"""Mixed-precision optimizer-update accounting tests."""

from __future__ import annotations

from eeg_cgdr.training import scaler_optimizer_step_succeeded


class _FakeScaler:
    def __init__(self, scales: list[float]) -> None:
        self._scales = iter(scales)
        self._scale = float(next(self._scales))
        self.step_calls = 0

    def get_scale(self) -> float:
        return self._scale

    def step(self, _optimizer: object) -> None:
        self.step_calls += 1

    def update(self) -> None:
        self._scale = float(next(self._scales))


def test_unchanged_or_growing_scale_counts_a_successful_optimizer_step() -> None:
    assert scaler_optimizer_step_succeeded(_FakeScaler([8.0, 8.0]), object())
    assert scaler_optimizer_step_succeeded(_FakeScaler([8.0, 16.0]), object())


def test_reduced_scale_marks_amp_overflow_skip() -> None:
    assert not scaler_optimizer_step_succeeded(
        _FakeScaler([8.0, 4.0]), object()
    )
