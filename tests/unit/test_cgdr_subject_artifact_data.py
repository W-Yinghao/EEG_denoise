"""Low-harness contracts for subject-artifact SGEYESUB preparation.

Real signal validation is performed by the aggregate CPU Slurm job through
``validate_real_subject_artifact_inputs``; these tests use deterministic
arrays only for API and leakage boundaries.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import eeg_cgdr.experiments.subject_artifact_data as data
from eeg_cgdr.experiments.sgeyesub_diffusion import TrialWindowOrigin


def _frozen_25() -> dict[str, object]:
    development = [
        {
            "study": "study01" if index < 5 else "study03",
            "layout_id": "layout01" if index < 5 else "layout03",
            "sampling_rate_hz": 200,
            "eeg_channels": 58 if index < 5 else 64,
        }
        for index in range(10)
    ]
    evaluation = []
    for index in range(15):
        study = ("study02", "study04", "study05")[index // 5]
        evaluation.append(
            {
                "study": study,
                "layout_id": {
                    "study02": "layout02",
                    "study04": "layout04",
                    "study05": "layout05",
                }[study],
                "sampling_rate_hz": 100 if study == "study04" else 200,
                "eeg_channels": 61 if study == "study04" else 64,
            }
        )
    return {"split": {"development_folds": development, "evaluation_folds": evaluation}}


def test_all_twenty_five_old_folds_are_mapped_to_development_indices() -> None:
    frozen = _frozen_25()
    assert data._unified_fold_route(frozen, 0) == ("development", 0)
    assert data._unified_fold_route(frozen, 9) == ("development", 9)
    assert data._unified_fold_route(frozen, 10) == ("evaluation", 0)
    assert data._unified_fold_route(frozen, 24) == ("evaluation", 14)


def test_runtime_context_keeps_three_eog_columns_at_retained_rank_two() -> None:
    raw = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.4, -0.3, 0.8]],
        dtype=np.float64,
    )
    left, singular, right = np.linalg.svd(raw, full_matrices=False)
    retained = left[:, :2] @ np.diag(singular[:2]) @ right[:2]
    scale = np.linalg.norm(retained, axis=0)
    normalized = retained / scale[None, :]
    context = data.RuntimeArtifactContext(
        role="matching",
        context_id="study05/p01:matching",
        raw_transfer=raw,
        full_transfer=retained,
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=singular,
        rank=2,
        projector=left[:, :2] @ left[:, :2].T,
        rho=0.75,
        calibration_duration_seconds=30.0,
        fit_recording_keys=("study05/p01",),
    )
    assert context.full_transfer.shape == (3, 3)
    assert context.rank == 2
    assert np.linalg.matrix_rank(context.full_transfer) == 2
    assert not np.allclose(context.full_transfer[:, 2], 0.0)
    assert context.model_kwargs()["full_transfer"] is context.full_transfer


def test_support_artifact_origin_is_resolved_without_query_eog() -> None:
    support_eog = np.arange(2 * 24, dtype=np.float64).reshape(2, 24)
    origin = TrialWindowOrigin("study01/p01", 1, 2, 6)
    value = data._artifact_window(support_eog, origin, samples_per_trial=8)
    np.testing.assert_array_equal(value, support_eog[:, 10:14])
    sealed = data.SealedQueryWindows(
        recording_key="study01/p01",
        observed=np.ones((2, 3, 8), dtype=np.float32),
        valid_time_mask=np.ones((2, 8), dtype=bool),
        origins=(
            TrialWindowOrigin("study01/p01", 0, 0, 8),
            TrialWindowOrigin("study01/p01", 1, 0, 8),
        ),
    )
    assert sealed.annotations_sealed is True
    assert not hasattr(sealed, "external_eog")
    assert not hasattr(sealed, "artifactclasses")


def test_outer_training_latent_normalizer_round_trip() -> None:
    normalizer = data.OuterTrainingLatentNormalizer(
        mean=np.asarray([1.0, -2.0]),
        standard_deviation=np.asarray([2.0, 4.0]),
        training_recording_keys=("study01/p01", "study01/p02"),
    )
    latent = np.asarray([[[1.0, 3.0], [-2.0, 2.0]]])
    np.testing.assert_allclose(
        normalizer.inverse_transform(normalizer.transform(latent)),
        latent,
    )


def test_support_rho_uses_only_frozen_singular_ratio_mapping() -> None:
    config = {
        "calibration": {
            "support_only_reliability": {"reference_singular_ratio": 0.25}
        }
    }
    transfer = SimpleNamespace(
        singular_values=np.asarray([4.0, 0.5, 0.1]),
        rank=2,
    )
    assert data._support_rho(config, transfer) == 0.5
    transfer = SimpleNamespace(
        singular_values=np.asarray([4.0, 2.0, 0.1]),
        rank=2,
    )
    assert data._support_rho(config, transfer) == 1.0


def test_real_validator_selects_five_cells_and_keeps_annotations_sealed(
    monkeypatch,
) -> None:
    frozen = _frozen_25()
    monkeypatch.setattr(data, "_load_frozen_config", lambda _config: frozen)

    def fake_prepare(_config, index):
        entries = tuple(frozen["split"]["development_folds"]) + tuple(
            frozen["split"]["evaluation_folds"]
        )
        entry = entries[index]
        training = (f"{entry['study']}/train",)
        heldout = (f"{entry['study']}/heldout",)
        return SimpleNamespace(
            fold=SimpleNamespace(
                fold_id=f"fold{index:02d}",
                study=entry["study"],
                layout_id=entry["layout_id"],
                training_recording_keys=training,
                heldout_recording_keys=heldout,
            ),
            model_dimensions=SimpleNamespace(
                eeg_channels=entry["eeg_channels"],
                eog_coordinates=2,
            ),
            training=SimpleNamespace(
                observed=np.zeros((3, entry["eeg_channels"], 8)),
                standardized_artifact_latent=np.zeros((3, 2, 8)),
            ),
            heldout={
                heldout[0]: SimpleNamespace(
                    matching=SimpleNamespace(rho=0.625),
                    query=SimpleNamespace(
                        observed=np.zeros((2, entry["eeg_channels"], 8)),
                        annotations_sealed=True,
                    )
                )
            },
        )

    monkeypatch.setattr(data, "prepare_subject_artifact_fold", fake_prepare)
    result = data.validate_real_subject_artifact_inputs({})
    assert result["status"] == "success_real_subject_artifact_inputs_sealed"
    assert result["frozen_fold_count"] == 25
    assert result["representative_exact_cell_count"] == 5
    assert result["query_annotations_opened"] is False
    assert result["support_rho_range"] == [0.625, 0.625]
    assert all(row["annotations_sealed"] for row in result["representatives"])
