"""Matched one-step clean predictor for V28."""
from __future__ import annotations

from torch import Tensor, nn

from eeg_scad.models.support_clean_cdm import CleanConditionalBackbone


class SupportCleanDET(nn.Module):
    forbidden_fields = ("query_EOG", "query_operator", "query_event", "subject_ID")

    def __init__(self, width: int = 64) -> None:
        super().__init__(); self.backbone=CleanConditionalBackbone(False,width)

    def forward(self, y: Tensor, context: Tensor) -> Tensor:
        return self.backbone(y,context)


__all__=["SupportCleanDET"]
