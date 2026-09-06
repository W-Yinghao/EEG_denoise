from pathlib import Path

from eeg_cgdr.data.bci2a_v10 import loso_manifest
from eeg_cgdr.experiments.bci2a_hierarchical_score_v10 import _distance, _support_query_ranges

import numpy as np
import torch
from torch import nn

from eeg_cgdr.models.hierarchical_score_lora import HierarchicalLoRAConv1d
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig,
    ArtifactSubspaceDiffusion,
)


def test_loso_manifest_is_frozen_and_subject_disjoint() -> None:
    rows = loso_manifest()
    assert len(rows) == 9
    for row in rows:
        heldout = int(row["heldout_subject"])
        training = {int(v) for v in str(row["outer_training_subjects"]).split(";")}
        assert len(training) == 8 and heldout not in training
        assert int(row["outcomes_opened_for_split"]) == 0


def test_v10_paths_do_not_point_to_sealed_mobile_data() -> None:
    text = Path("configs/cgdr/bci2a_hierarchical_score_v10.yaml").read_text()
    assert "mobile" not in text.lower()
    assert "BCI-IV" in text


def test_transfer_distance_is_zero_only_for_identical_operator() -> None:
    base = np.arange(24, dtype=float).reshape(4, 2, 3) + 1
    assert max(_distance(base, base)) < 1.0e-12
    assert _distance(base, base * np.linspace(1, 2, 4)[:, None, None])[0] > 0


def test_support_query_ranges_have_five_second_guard() -> None:
    support, query = _support_query_ranges([(100.0, "768")], 50_000, 250.0)
    assert support.stop == 95 * 250
    assert query.start == 100 * 250


def test_hierarchical_alpha_zero_is_exact_population() -> None:
    base=nn.Conv1d(3,5,3,padding=1);layer=HierarchicalLoRAConv1d(base);x=torch.randn(2,3,16)
    population=layer(x);layer.set_alpha(torch.zeros(2,4));assert torch.equal(population,layer(x))
    with torch.no_grad():layer.up[0].weight.fill_(.1)
    layer.set_alpha(torch.tensor([[.5,0,0,0],[.5,0,0,0]]));assert not torch.equal(population,layer(x))


def test_diffusion_loss_masks_inactive_rank_coordinate() -> None:
    model = ArtifactSubspaceDiffusion(
        ArtifactSubspaceConfig(eeg_channels=2, signal_length=8)
    ).eval()
    observed = torch.randn(1, 2, 8)
    condition = {
        "observed": observed,
        "basis": torch.eye(2)[None],
        "reliability": torch.ones(1),
        "rank_mask": torch.tensor([[True, False]]),
        "valid_time_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    first = torch.randn(1, 2, 8)
    second = first.clone()
    second[:, 1] += 1000.0
    timestep = torch.tensor([500])
    noise = torch.randn_like(first)
    generator = torch.Generator().manual_seed(5)
    loss_a = model.training_loss(
        first, generator=generator, timestep=timestep, noise=noise, **condition
    )[0]
    loss_b = model.training_loss(
        second, generator=generator, timestep=timestep, noise=noise, **condition
    )[0]
    assert torch.equal(loss_a, loss_b)
