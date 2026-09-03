"""Deformation quality-control gates (plan §3.4).

Every generated transformation must pass these checks before it is used in any
confirmatory cell. A failed transformation is regenerated (different sigma/seed) or
excluded under a recorded, condition-blind rule -- the regenerate/exclude decision is
made by the orchestrator, but the checks and the resulting audit record live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deformation import DeformationField, jacobian_determinant, realized_volume_change_pct


@dataclass(frozen=True)
class DeformationQCResult:
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]


def check_positive_jacobian(jacobian_det: np.ndarray) -> tuple[bool, float]:
    """Plan §3.4: 'positive Jacobian determinant everywhere' -- a non-positive
    determinant means the transformation folds space and is not a valid diffeomorphism.
    """
    minimum = float(np.min(jacobian_det))
    return minimum > 0.0, minimum


def check_realized_amplitude(
    realized_pct: float,
    requested_pct: float,
    *,
    tolerance_pct: float,
) -> tuple[bool, float]:
    """Plan §3.4: 'realized target-volume change within a predefined tolerance of the
    requested change.'
    """
    diff = abs(realized_pct - requested_pct)
    return diff <= tolerance_pct, diff


def check_negligible_outside_target(
    jacobian_det: np.ndarray,
    transition_band_mask: np.ndarray,
    *,
    max_abs_change_pct: float,
) -> tuple[bool, float]:
    """Plan §3.4: 'negligible change outside the target and transition band.'

    `transition_band_mask` marks every voxel inside the target region plus its
    intended taper -- everywhere outside that mask must show near-zero apparent
    volume change.
    """
    outside = ~np.asarray(transition_band_mask, dtype=bool)
    if not outside.any():
        return True, 0.0
    change_pct = 100.0 * (jacobian_det[outside].astype(np.float64) - 1.0)
    worst = float(np.max(np.abs(change_pct)))
    return worst <= max_abs_change_pct, worst


def check_boundary_smoothness(
    jacobian_det: np.ndarray,
    *,
    max_local_gradient: float,
) -> tuple[bool, float]:
    """Plan §3.4: 'smooth transition at the target boundary.'

    A hard edge in the deformation shows up as a large spatial gradient of the
    Jacobian determinant field; bound its magnitude everywhere.
    """
    gradients = np.gradient(jacobian_det.astype(np.float64))
    magnitude = np.sqrt(sum(g**2 for g in gradients))
    worst = float(np.max(magnitude))
    return worst <= max_local_gradient, worst


def check_grid_identity(
    baseline_shape: tuple[int, ...],
    followup_shape: tuple[int, ...],
) -> tuple[bool, bool]:
    """Plan §3.4: 'identical image grid and metadata where required by the comparison.'"""
    match = tuple(baseline_shape) == tuple(followup_shape)
    return match, match


def evaluate_deformation_qc(
    field: DeformationField,
    *,
    region_mask: np.ndarray,
    transition_band_mask: np.ndarray,
    requested_change_pct: float,
    tolerance_pct: float,
    max_outside_change_pct: float,
    max_boundary_gradient: float,
    baseline_shape: tuple[int, int, int],
    followup_shape: tuple[int, int, int],
) -> DeformationQCResult:
    """Run every plan §3.4 check on one deformation and return a pass/fail record.

    Does not decide whether to regenerate or exclude a failing transformation --
    that policy belongs to the orchestrator, which can log this result under the
    condition-blind rule plan §3.4 requires.
    """
    jacobian_det = jacobian_determinant(field)
    realized_pct = realized_volume_change_pct(jacobian_det, region_mask)

    positive_pass, min_jacobian = check_positive_jacobian(jacobian_det)
    amplitude_pass, amplitude_diff = check_realized_amplitude(
        realized_pct, requested_change_pct, tolerance_pct=tolerance_pct,
    )
    outside_pass, outside_worst = check_negligible_outside_target(
        jacobian_det, transition_band_mask, max_abs_change_pct=max_outside_change_pct,
    )
    smooth_pass, smooth_worst = check_boundary_smoothness(
        jacobian_det, max_local_gradient=max_boundary_gradient,
    )
    grid_pass, _ = check_grid_identity(baseline_shape, followup_shape)

    checks = {
        "positive_jacobian": positive_pass,
        "realized_amplitude_within_tolerance": amplitude_pass,
        "negligible_outside_target": outside_pass,
        "boundary_smooth": smooth_pass,
        "grid_identical": grid_pass,
    }
    values = {
        "min_jacobian": min_jacobian,
        "realized_change_pct": realized_pct,
        "amplitude_abs_diff_pct": amplitude_diff,
        "max_outside_change_pct": outside_worst,
        "max_boundary_gradient": smooth_worst,
    }
    return DeformationQCResult(passed=all(checks.values()), checks=checks, values=values)
