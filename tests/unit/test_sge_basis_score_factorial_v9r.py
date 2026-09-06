from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eeg_cgdr.experiments.sge_basis_score_factorial_v9r import CAL_WRONG, RAW_WRONG, _coverage_from_rows
from eeg_cgdr.models.artifact_subspace_diffusion import ArtifactSubspaceDiffusion, DeterministicSubspaceEstimator
from eeg_cgdr.models.adaptation_replay import AdaptationReplay
from eeg_cgdr.models.artifact_subspace_score_lora import LoRAConv1d


def test_raw_and_calibrated_wrong_names_are_disjoint()->None:
    assert RAW_WRONG.fullmatch("DIFF-D11-WRONG-BOTH-0")
    assert not RAW_WRONG.fullmatch("DIFF-D11-WRONG-BOTH-0-CAL")
    assert CAL_WRONG.fullmatch("DIFF-D11-WRONG-BOTH-0-CAL")
    assert not CAL_WRONG.fullmatch("DIFF-D11-WRONG-BOTH-0")


def test_adaptation_replay_roundtrip(tmp_path:Path)->None:
    replay=AdaptationReplay.create(seed=9,pair_count=7,validation_count=5,updates=10,batch_size=4,timesteps=1000,signal_length=16,checkpoint_steps=(0,10))
    path=tmp_path/"replay.npz";replay.save(path);loaded=AdaptationReplay.load(path);loaded.validate(pair_count=7,validation_count=5,signal_length=16)
    np.testing.assert_array_equal(replay.minibatch_indices,loaded.minibatch_indices)
    np.testing.assert_array_equal(replay.gaussian_noise,loaded.gaussian_noise)
    np.testing.assert_array_equal(replay.inference_noise_bank,loaded.inference_noise_bank)


def test_zero_step_lora_is_backbone_equivalent()->None:
    base=torch.nn.Conv1d(3,4,3,padding=1);value=torch.randn(2,3,19);expected=base(value).detach();lora=LoRAConv1d(base,rank=2)
    torch.testing.assert_close(lora(value),expected)


def test_coverage_flags_are_explicit_integers()->None:
    rows=[{"candidate":"D11","personalization_eligible":0},{"candidate":"D11","personalization_eligible":1},{"candidate":"D10","personalization_eligible":1}]
    assert _coverage_from_rows(rows)==(1,2,.5)


def test_query_inference_contract_excludes_evaluator_fields()->None:
    forbidden={"query_EOG","query_labels","query_outcomes","clean_EEG"}
    assert forbidden.isdisjoint(ArtifactSubspaceDiffusion.visible_input_fields)
    assert forbidden.isdisjoint(DeterministicSubspaceEstimator.visible_input_fields)
