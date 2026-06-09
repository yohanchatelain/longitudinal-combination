"""Aggregation functions: collapse per-timepoint feature arrays into one vector per subject.

All functions accept pre-z-scored features (scaling happens inside CV folds).
All pure functions — no side effects.

Arm A (state target — absolute age):
  mean, concatenation, annualized_rate, lme_slope

Arm B (change target — brain change rate = PC1 of annualized FS ROI feature change):
  annualized_rate, difference, lme_slope_change  (expected to win)
  mean, concatenation                             (negative controls — state descriptors)
  NOTE: FS ROI results in Arm B carry leakage_warning=True (target derived from FS ROI).

Arm B2 (change target — annualized eTIV rate = (eTIV_last - eTIV_first) / delta_t):
  difference, lme_slope_change, annualized_rate   (expected to win)
  mean, concatenation                             (negative controls)
  NOTE: FS ROI results in Arm B2 carry leakage_warning=True (eTIV correlates with FS ROI features).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Arm-A aggregations
# ---------------------------------------------------------------------------

def mean(features: np.ndarray) -> np.ndarray:
    """Element-wise mean across all available timepoints.

    Args:
        features: (n_timepoints, n_features)
    Returns:
        (n_features,)
    """
    return features.mean(axis=0)


def concatenation(
    features: np.ndarray,
    t_first_local: int,
    t_mid_local: int,
    t_last_local: int,
) -> np.ndarray:
    """[f(t_first), f(t_mid), f(t_last)] concatenation.

    Args:
        features:      (n_timepoints, n_features) — rows are sorted timepoints
        t_first_local: local index (row in features) of first timepoint
        t_mid_local:   local index of sampled mid timepoint
        t_last_local:  local index of last timepoint
    Returns:
        (3 * n_features,)
    """
    return np.concatenate(
        [features[t_first_local], features[t_mid_local], features[t_last_local]]
    )


def annualized_rate(features: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """(f_last - f_first) / (age_last - age_first).

    Valid for both Arm A and Arm B. Requires delta_t > 0 (subjects with
    delta_t == 0 are excluded by the exclude_arm_b manifest flag).

    Args:
        features: (n_timepoints, n_features)
        ages:     (n_timepoints,) aligned with features
    Returns:
        (n_features,)
    """
    delta_age = float(ages[-1] - ages[0])
    if abs(delta_age) < 1e-8:
        raise ValueError(
            f"annualized_rate: delta_age={delta_age:.6f} is near zero — "
            "cannot compute rate. Exclude subjects with delta_t=0."
        )
    return (features[-1] - features[0]) / delta_age


def lme_slope(subject_id, lme_subjects: np.ndarray, lme_slopes: np.ndarray) -> np.ndarray:
    """Return the BLUP slope vector for a subject from a fitted LME estimator.

    Args:
        subject_id:   the subject to look up
        lme_subjects: (n_subjects,) subject ID array returned by lme.transform()
        lme_slopes:   (n_subjects, n_features) slope matrix returned by lme.transform()
    Returns:
        (n_features,)
    """
    idx = np.where(lme_subjects == subject_id)[0]
    if len(idx) == 0:
        raise KeyError(f"Subject '{subject_id}' not found in LME transform output.")
    return lme_slopes[idx[0]]


# ---------------------------------------------------------------------------
# Arm-B aggregations (change descriptors only)
# ---------------------------------------------------------------------------

def difference(features: np.ndarray) -> np.ndarray:
    """f(t_last) - f(t_first).  No division by time — anti-circularity rule.

    Args:
        features: (n_timepoints, n_features) — rows sorted by timepoint
    Returns:
        (n_features,)
    """
    return features[-1] - features[0]


def lme_slope_change(
    subject_id: str, lme_subjects: np.ndarray, lme_slopes: np.ndarray
) -> np.ndarray:
    """Slope-reconstructed change descriptor: same BLUP slope as lme_slope.

    Used in Arm B as a regularized change input. The slope is NOT multiplied
    by the subject's own Δt — that would re-introduce circularity.

    Returns:
        (n_features,) — the subject's BLUP slope vector
    """
    return lme_slope(subject_id, lme_subjects, lme_slopes)


# ---------------------------------------------------------------------------
# Registry: maps aggregation name → callable metadata
# ---------------------------------------------------------------------------

# Specification for each aggregation used in the factorial experiment.
# 'arms':       set of arm letters where this aggregation is valid
# 'needs_lme':  whether a fitted LME estimator is required
# 'needs_ages': whether age information is required (annualized_rate)
# 'cohort':     'all' (≥2 tp) or 'concat' (≥3 tp required)
# 'is_neg_ctrl': True if this is a negative control in Arm B
AGGREGATION_SPECS: dict[str, dict] = {
    "mean":              {"arms": {"A", "B", "B2"}, "needs_lme": False, "needs_ages": False, "cohort": "all",    "is_neg_ctrl_B": True,  "is_neg_ctrl_B2": True},
    "concatenation":     {"arms": {"A", "B", "B2"}, "needs_lme": False, "needs_ages": False, "cohort": "concat", "is_neg_ctrl_B": True,  "is_neg_ctrl_B2": True},
    "annualized_rate":   {"arms": {"A", "B", "B2"}, "needs_lme": False, "needs_ages": True,  "cohort": "all",    "is_neg_ctrl_B": False, "is_neg_ctrl_B2": False},
    "lme_slope":         {"arms": {"A"},             "needs_lme": True,  "needs_ages": False, "cohort": "all",    "is_neg_ctrl_B": False, "is_neg_ctrl_B2": False},
    "difference":        {"arms": {"B", "B2"},       "needs_lme": False, "needs_ages": False, "cohort": "all",    "is_neg_ctrl_B": False, "is_neg_ctrl_B2": False},
    "lme_slope_change":  {"arms": {"B", "B2"},       "needs_lme": True,  "needs_ages": False, "cohort": "all",    "is_neg_ctrl_B": False, "is_neg_ctrl_B2": False},
}


def validate_arm(aggregation_name: str, arm: str) -> None:
    """Raise ValueError if the aggregation is not valid for this arm."""
    spec = AGGREGATION_SPECS.get(aggregation_name)
    if spec is None:
        raise ValueError(f"Unknown aggregation: '{aggregation_name}'")
    if arm.upper() not in spec["arms"]:
        raise ValueError(
            f"Aggregation '{aggregation_name}' is not valid for Arm {arm}. "
            f"Valid arms: {spec['arms']}."
        )
