from __future__ import annotations

from pathlib import Path

import inspect
import torch
import yaml

from eeg_cgdr.experiments.literature_guided_v3 import PROTOCOL, _shuffled_support, task_rows
from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusionConfig
from eeg_cgdr.models.literature_guided_v3 import (
    DirectSupportAdapter,
    RawSupportConfig,
    RawSupportSetEncoder,
    RawSupportTokenDeterministic,
    RawSupportTokenDiffusion,
    SupportStatisticControl,
    discrete_selector_features,
)


def test_v3_config_freezes_exploratory_scope_and_no_query_external_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/cgdr/literature_guided_exploration_v3/exploration.yaml").read_text()
    )
    assert config["protocol_id"] == PROTOCOL
    assert config["harness_level"] == 1
    assert config["scientific_role"] == "full_real_development_screen_not_confirmation"
    assert config["screen"]["query_external_signals_inference"] is False
    assert config["training"]["maximum_gpu_concurrency"] == 8


def test_v3_task_list_is_full_one_seed_route_screen() -> None:
    rows = task_rows()
    assert len(rows) == 3 * 26
    assert {row["route"] for row in rows} == {
        "P_A_RAW_SUPPORT_TOKENS",
        "P_B_DIRECT_SUPPORT_ADAPTER",
        "P_D_SUPPORT_STAT_CONTROL",
    }
    assert {row["seed"] for row in rows} == {20260811}
    for route in {row["route"] for row in rows}:
        selected = [row for row in rows if row["route"] == route]
        assert sum(row["dataset"] == "klados" for row in selected) == 1
        assert sum(row["dataset"] == "sgeyesub" for row in selected) == 25


def _support_fixture() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(77)
    eeg = torch.randn(2, 4, 8, 32, generator=generator)
    latent = torch.randn(2, 4, 2, 32, generator=generator)
    mask = torch.ones(2, 4, 32, dtype=torch.bool)
    present = torch.ones(2)
    return eeg, latent, mask, present


def test_raw_support_encoder_is_permutation_invariant_and_uses_support() -> None:
    eeg, latent, mask, present = _support_fixture()
    encoder = RawSupportSetEncoder(8, 2, RawSupportConfig(token_width=32, token_count=2, encoder_layers=1))
    encoder.eval()
    _, first = encoder(eeg, latent, mask, present)
    order = torch.tensor([2, 0, 3, 1])
    _, permuted = encoder(eeg[:, order], latent[:, order], mask[:, order], present)
    assert torch.allclose(first, permuted, atol=1e-6, rtol=1e-6)
    _, wrong = encoder(eeg.flip(0), latent.flip(0), mask, present)
    assert float((first - wrong).abs().max()) > 1e-5
    _, absent = encoder(eeg, latent, mask, torch.zeros_like(present))
    assert torch.count_nonzero(absent) == 0


def test_raw_support_models_share_target_information_and_forbid_query_eog() -> None:
    model = ArtifactLatentModelConfig(
        eeg_channels=8, latent_channels=2, signal_length=32,
        base_channels=8, groupnorm_groups=4, time_sinusoidal_dim=16,
        time_embed_dim=32, attention_length=4, attention_heads=2,
    )
    support = RawSupportConfig(token_width=32, token_count=2, encoder_layers=1)
    diffusion = RawSupportTokenDiffusion(
        model, ArtifactLatentDiffusionConfig(num_timesteps=1000), support
    )
    deterministic = RawSupportTokenDeterministic(model, support)
    for callable_value in (diffusion.predict_v, diffusion.training_loss, diffusion.latent_samples, deterministic.forward):
        fields = set(inspect.signature(callable_value).parameters)
        assert not fields.intersection({"query_EOG", "query_artifact_label", "query_outcome", "participant_ID"})


def test_direct_adapter_is_identity_at_initialization() -> None:
    value = torch.randn(3, 2, 32)
    mask = torch.ones(3, 32, dtype=torch.bool)
    adapter = DirectSupportAdapter(2, rank=2)
    assert torch.equal(adapter(value, mask), value)


def test_support_statistic_control_uses_one_statistic_for_query_and_support() -> None:
    support = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]]
    )
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    query = torch.tensor([[[2.5, 3.5], [6.5, 7.5]]])
    control = SupportStatisticControl()
    normalized_query = control(query, support, mask)
    normalized_support = control.normalize_support(support, mask)
    assert normalized_query.shape == query.shape
    assert normalized_support.shape == support.shape
    assert torch.allclose(normalized_support.mean(dim=(1, 3)), torch.zeros(1, 2), atol=1e-6)


def test_selector_features_use_only_eeg_outputs_and_support_quantiles() -> None:
    query = torch.randn(2, 8, 32)
    population = query * 0.9
    matching = query * 0.8
    samples = torch.stack((matching, matching + 0.01, matching - 0.01))
    quantiles = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
    features = discrete_selector_features(query, population, matching, samples, quantiles)
    assert features.shape == (2, 7)
    assert torch.isfinite(features).all()


def test_shuffled_support_changes_only_canonical_support_pairing() -> None:
    eeg = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 8).numpy()
    latent = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 8).numpy()
    valid = torch.ones(4, 8, dtype=torch.bool).numpy()
    shuffled_eeg, shuffled_latent, shuffled_valid = _shuffled_support((eeg, latent, valid))
    assert (shuffled_eeg == eeg).all()
    assert (shuffled_valid == valid).all()
    assert not (shuffled_latent == latent).all()
