"""Real EEG loaders and split construction for CGDR."""

from .eye_bci import (
    DEFAULT_EYE_BCI_TARGETS,
    EYE_BCI_ROOT,
    EyeBciBoundedRecord,
    EyeBciTarget,
    read_default_eye_bci_targets,
    read_eye_bci_target,
)
from .klados import KladosRecord, analyze_klados_metadata, load_klados_records
from .pipeline import (
    CalibrationWindow,
    CleanNormalizer,
    GroupedKladosSource,
    KladosPipeline,
    PopulationWindow,
    QueryWindow,
    assert_calibration_query_disjoint,
    build_klados_pipeline,
    fit_outer_train_clean_normalizer,
    group_klados_records,
    window_grouped_klados,
)

__all__ = [
    "CalibrationWindow",
    "CleanNormalizer",
    "DEFAULT_EYE_BCI_TARGETS",
    "EYE_BCI_ROOT",
    "EyeBciBoundedRecord",
    "EyeBciTarget",
    "GroupedKladosSource",
    "KladosPipeline",
    "KladosRecord",
    "PopulationWindow",
    "QueryWindow",
    "analyze_klados_metadata",
    "assert_calibration_query_disjoint",
    "build_klados_pipeline",
    "fit_outer_train_clean_normalizer",
    "group_klados_records",
    "load_klados_records",
    "read_default_eye_bci_targets",
    "read_eye_bci_target",
    "window_grouped_klados",
]
