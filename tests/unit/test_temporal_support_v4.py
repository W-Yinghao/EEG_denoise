from __future__ import annotations

import inspect
import torch

from eeg_cgdr.models.temporal_support_conditioner import TemporalSupportCleaner
from eeg_cgdr.models.temporal_support_diffusion import TemporalDiffusionConfig, TemporalSupportCorrectionDiffusion, cosine_alpha_bar


def test_temporal_support_preserves_order_and_changes_output() -> None:
    torch.manual_seed(7); model=TemporalSupportCleaner(eeg_channels=4,imu_channels=2,eog_channels=1,width=32).eval()
    # Make the initially-zero output head observable for this structural test.
    torch.nn.init.normal_(model.output[-1].weight,std=0.01)
    query=torch.randn(2,4,64); eeg=torch.randn(2,4,256); imu=torch.randn(2,2,256); eog=torch.randn(2,1,256)
    present=torch.ones(2,3); context=torch.ones(2)
    first=model(query,support_eeg=eeg,support_imu=imu,support_eog=eog,modality_present=present,context_present=context)
    reversed_value=model(query,support_eeg=eeg.flip(-1),support_imu=imu.flip(-1),support_eog=eog.flip(-1),modality_present=present,context_present=context)
    assert first.shape==query.shape and float((first-reversed_value).abs().max())>1e-7


def test_temporal_model_surface_forbids_query_side_modalities_and_identity() -> None:
    fields=set(inspect.signature(TemporalSupportCleaner.forward).parameters)
    assert not fields.intersection({"query_EOG","query_IMU","query_event_label","participant_ID","query_outcome"})


def test_temporal_diffusion_schedule_and_common_context_shape() -> None:
    schedule=cosine_alpha_bar(1000);assert schedule.shape==(1000,) and schedule[-1]<1e-4
    model=TemporalSupportCorrectionDiffusion(TemporalDiffusionConfig(timesteps=32,ddim_steps=4,posterior_samples=2))
    assert not set(model.denoiser.forbidden_inputs).intersection(set(inspect.signature(model.denoiser.forward).parameters))
