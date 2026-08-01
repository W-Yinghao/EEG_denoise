"""Population and calibration-guided diffusion inference."""

from .one_step import InformationMatchedOneStep, OneStepResult
from .population import (
    FullVJPResult,
    GuidanceStabilityConfig,
    GuidanceStepTrace,
    PopulationObservationState,
    PopulationOnlyInference,
)
from .sampler_candidates import (
    SAMPLER_CANDIDATES,
    RepairedSamplerRunner,
    SamplerCandidateSpec,
    SamplerMechanism,
    SamplerRunResult,
    sampler_candidate,
)
from .states import (
    CalibrationContextProjector,
    DatasetPopulationProjector,
    attenuation_from_external_reference,
    dataset_population_and_context_states,
    dataset_population_state,
    frame_attenuation_from_external_reference,
    matched_population_and_context_states,
    population_state_only,
    rho_interpolated_precision_state,
)

__all__ = [
    "PopulationObservationState",
    "PopulationOnlyInference",
    "FullVJPResult",
    "GuidanceStabilityConfig",
    "GuidanceStepTrace",
    "OneStepResult",
    "SamplerMechanism",
    "SamplerCandidateSpec",
    "SAMPLER_CANDIDATES",
    "RepairedSamplerRunner",
    "SamplerRunResult",
    "sampler_candidate",
    "CalibrationContextProjector",
    "DatasetPopulationProjector",
    "attenuation_from_external_reference",
    "frame_attenuation_from_external_reference",
    "dataset_population_state",
    "dataset_population_and_context_states",
    "rho_interpolated_precision_state",
    "matched_population_and_context_states",
    "population_state_only",
    "InformationMatchedOneStep",
]
