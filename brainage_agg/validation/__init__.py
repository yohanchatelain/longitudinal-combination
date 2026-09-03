"""Preregistered positive-control validation for voxel attribution."""

from .gates import evaluate_decision_gates
from .injection import (
    apply_multiplicative_attenuation,
    deterministic_subject_assignment,
    select_calibrated_amplitude,
    tapered_label_mask,
)
from .metrics import localization_metrics
from .statistics import empirical_region_pvalues, maxT_pseudo_null_calibration

__all__ = [
    "apply_multiplicative_attenuation",
    "deterministic_subject_assignment",
    "empirical_region_pvalues",
    "evaluate_decision_gates",
    "localization_metrics",
    "maxT_pseudo_null_calibration",
    "select_calibrated_amplitude",
    "tapered_label_mask",
]
