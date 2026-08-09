import numpy as np
import torch

from eeg_cgdr.experiments.raw_support_clean_diffusion import (
    _karcher_mean,
    _log_barycenter,
    _robust_correlation_geometry,
    _sign_flip,
)
from eeg_cgdr.models.raw_support_clean_diffusion import (
    DeterministicRawSupportCleaner,
    RawSupportCleanConfig,
    RawSupportCleanDiffusion,
)


def test_support_set_permutation_invariance_and_context_response():
    torch.manual_seed(4)
    config = RawSupportCleanConfig(base_channels=16)
    model = DeterministicRawSupportCleaner(config).eval()
    query = torch.randn(2, 3, 512)
    support = torch.randn(2, 16, 3, 500)
    with torch.no_grad():
        reference = model(query_y=query, support_eeg=support)
        permuted = model(query_y=query, support_eeg=support[:, torch.randperm(16)])
        changed = model(query_y=query, support_eeg=support.flip(-1))
    assert torch.allclose(reference, permuted, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(reference, changed)


def test_det_and_diff_use_exact_same_active_parameterization():
    config = RawSupportCleanConfig(base_channels=16)
    det = DeterministicRawSupportCleaner(config)
    diff = RawSupportCleanDiffusion(config)
    assert sum(p.numel() for p in det.parameters()) == sum(p.numel() for p in diff.parameters())
    fields = {"query_y": torch.randn(2, 3, 512), "support_eeg": torch.randn(2, 16, 3, 500)}
    loss = det(**fields).square().mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in det.parameters())


def test_forbidden_query_fields_and_sampler_replay():
    config = RawSupportCleanConfig(base_channels=16)
    model = RawSupportCleanDiffusion(config).eval()
    assert "query_EOG" in model.forbidden_fields
    assert "query_clean_target" in model.forbidden_fields
    query = torch.randn(1, 3, 512); support = torch.randn(1, 16, 3, 500); noise = torch.randn_like(query)
    with torch.no_grad():
        first = model.sample(query_y=query, support_eeg=support, initial_noise=noise)
        second = model.sample(query_y=query, support_eeg=support, initial_noise=noise)
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[..., 500:]) == 0


def test_true_spd_barycenters_and_participant_sign_flip():
    rng = np.random.default_rng(3)
    matrices = [_robust_correlation_geometry(rng.normal(size=(3, 1000))) for _ in range(4)]
    for mean in (_karcher_mean(matrices), _log_barycenter(matrices)):
        assert np.all(np.linalg.eigvalsh(mean) > 0)
    assert _sign_flip(np.ones(9), True) == 1 / 512
    assert _sign_flip(np.ones(9), False) == 2 / 512
