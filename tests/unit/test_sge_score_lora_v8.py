from __future__ import annotations

import numpy as np
import torch

from eeg_cgdr.experiments.sge_score_lora_v8 import _cluster_bootstrap, _operator_distances
from eeg_cgdr.models.artifact_subspace_diffusion import ArtifactSubspaceConfig, ArtifactSubspaceDiffusion
from eeg_cgdr.models.artifact_subspace_score_lora import inject_score_lora


def test_operator_distance_is_zero_in_common_coordinates() -> None:
    transfer = np.random.default_rng(2).normal(size=(4, 2, 7))
    eog = np.random.default_rng(3).normal(size=(2, 128))
    values = _operator_distances(transfer, transfer.copy(), eog)
    assert max(values.values()) < 1e-10


def test_cluster_bootstrap_uses_unit_rows() -> None:
    rows = []
    for fold in range(2):
        for participant in range(3):
            rows.append({"budget_seconds": 30.0, "study": "study01", "fold_cluster": fold,
                         "recording_key": f"p{fold}{participant}", "oracle_relative_improvement": .1,
                         "deployable_relative_improvement": .05, "match_relative_improvement": .02,
                         "match_vs_wrong_relative_improvement": .01})
    result = _cluster_bootstrap(rows, 20, 11)
    assert all(row["denominator"] == 6 for row in result)
    assert all(row["resampling"].startswith("study_stratified") for row in result)


def test_score_lora_is_internal_zero_initialized_and_rank_four() -> None:
    model = ArtifactSubspaceDiffusion(ArtifactSubspaceConfig(eeg_channels=4, signal_length=32, base_channels=8))
    model.eval()
    state = torch.randn(2, 2, 32); observed = torch.randn(2, 4, 32); basis, _ = torch.linalg.qr(torch.randn(2, 4, 2))
    condition = {"observed": observed, "basis": basis, "reliability": torch.ones(2), "rank_mask": torch.ones(2, 2, dtype=torch.bool), "valid_time_mask": torch.ones(2, 32, dtype=torch.bool)}
    timestep = torch.tensor([100, 700]); before = model.backbone(state, timestep, **condition)
    summary = inject_score_lora(model.backbone, rank=4)
    after = model.backbone(state, timestep, **condition)
    torch.testing.assert_close(after, before)
    assert summary.rank == 4 and summary.adapted_convolutions > 4 and summary.trainable_parameters > 0
    assert all((not parameter.requires_grad) for name, parameter in model.named_parameters() if ".base." in name)
