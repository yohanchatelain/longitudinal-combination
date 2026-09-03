"""Synthetic longitudinal deformation fields with exact Jacobian-derived ground truth.

Plan §3: every synthetic follow-up condition is a smooth, invertible displacement
field applied only to the follow-up. The realized Jacobian-determinant integral over
the target region -- not the requested nominal amplitude -- is the ground truth used
in analysis (plan §3.2); `qc.py` enforces that the realized value stays within
tolerance of what was requested before a transformation is accepted.

All fields are expressed as a forward map phi(x) = x + d(x) in voxel-index
coordinates, so percentage volume changes are exact regardless of voxel size.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as _dataclass_field

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DeformationField:
    """A displacement field d(x) defining the forward map phi(x) = x + d(x).

    `displacement` has shape (3, D, H, W): one component array per axis, in voxel
    units, matching the volume's array axes in order.
    """

    displacement: np.ndarray
    condition: str
    params: dict = _dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.displacement.ndim != 4 or self.displacement.shape[0] != 3:
            raise ValueError(
                f"displacement must have shape (3, D, H, W), got {self.displacement.shape}"
            )


def _index_grid(shape: tuple[int, int, int]) -> np.ndarray:
    return np.indices(shape, dtype=np.float32)


def _volume_center(shape: tuple[int, int, int]) -> np.ndarray:
    return (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0


def identity_field(shape: tuple[int, int, int]) -> DeformationField:
    """The exact-duplicate condition: zero displacement everywhere, ΔV_true = 0."""
    return DeformationField(
        displacement=np.zeros((3, *shape), dtype=np.float32),
        condition="exact_duplicate",
        params={},
    )


def rigid_subvoxel_field(
    shape: tuple[int, int, int],
    *,
    translation_voxels: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
) -> DeformationField:
    """The resampling-sham condition.

    A rigid (rotation + translation) transform has a Jacobian determinant of exactly
    1.0 everywhere, so this field's true anatomical change is exactly zero by
    construction -- independent of how much interpolation noise the resulting
    resample introduces. This isolates interpolation/registration sensitivity from
    any confound with real geometric signal (plan §3.1).
    """
    grid = _index_grid(shape)
    center = _volume_center(shape).reshape(3, 1, 1, 1)
    relative = grid - center
    rotation = Rotation.from_euler("xyz", rotation_degrees, degrees=True).as_matrix().astype(np.float32)
    rotated = np.einsum("ij,jxyz->ixyz", rotation, relative)
    translation = np.asarray(translation_voxels, dtype=np.float32).reshape(3, 1, 1, 1)
    displacement = (rotated - relative) + translation
    return DeformationField(
        displacement=displacement.astype(np.float32),
        condition="resampling_sham",
        params={"translation_voxels": tuple(translation_voxels), "rotation_degrees": tuple(rotation_degrees)},
    )


def radial_bump_field(
    shape: tuple[int, int, int],
    target_mask: np.ndarray,
    *,
    magnitude: float,
    sigma_voxels: float,
    direction: str = "contract",
) -> DeformationField:
    """A smooth, localized radial displacement field centered on `target_mask`'s centroid.

    d(x) = ±magnitude * (x - centroid) * exp(-||x - centroid||^2 / (2 sigma^2)):
    a Gaussian-weighted radial pull toward (`direction="contract"`) or push away from
    (`direction="expand"`) the target centroid. Smooth by construction; invertibility
    (positive Jacobian everywhere) must still be checked by `qc.py` for the chosen
    magnitude/sigma, not assumed.
    """
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    if sigma_voxels <= 0:
        raise ValueError("sigma_voxels must be positive")
    if direction not in ("contract", "expand"):
        raise ValueError(f"Unsupported direction: {direction!r}")
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != shape:
        raise ValueError(f"target_mask shape {target_mask.shape} != volume shape {shape}")
    if not target_mask.any():
        raise ValueError("target_mask must contain at least one voxel")

    centroid = np.asarray(ndi.center_of_mass(target_mask), dtype=np.float32).reshape(3, 1, 1, 1)
    grid = _index_grid(shape)
    offset = grid - centroid
    squared_distance = np.sum(offset**2, axis=0)
    bump = np.exp(-squared_distance / (2.0 * sigma_voxels**2)).astype(np.float32)
    sign = -1.0 if direction == "contract" else 1.0
    displacement = sign * magnitude * offset * bump[None]
    return DeformationField(
        displacement=displacement.astype(np.float32),
        condition=f"radial_{direction}",
        params={"magnitude": float(magnitude), "sigma_voxels": float(sigma_voxels), "direction": direction},
    )


def global_scaling_field(shape: tuple[int, int, int], *, scale: float) -> DeformationField:
    """Uniform isotropic scaling about the volume centroid.

    Exact sanity-check condition (plan §3.1, "end-to-end magnitude sanity check"):
    phi(x) = center + scale * (x - center) has Jacobian = scale * I everywhere, so
    ΔV_true = 100 * (scale**3 - 1) exactly -- no numerical estimation error, useful
    as a check on `jacobian_determinant` and `realized_volume_change_pct` themselves.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    center = _volume_center(shape).reshape(3, 1, 1, 1)
    grid = _index_grid(shape)
    relative = grid - center
    displacement = (scale - 1.0) * relative
    return DeformationField(
        displacement=displacement.astype(np.float32),
        condition="global_scaling",
        params={"scale": float(scale)},
    )


def jacobian_determinant(field: DeformationField) -> np.ndarray:
    """det(J_phi) at every voxel, where phi(x) = x + displacement(x).

    J_phi = I + grad(d), computed with central differences (np.gradient) per
    displacement component. Works identically for every condition type above; the
    exact conditions (rigid, global scaling) double as regression checks on this
    function since their true determinant is known in closed form.
    """
    displacement = field.displacement.astype(np.float64)
    jacobian = np.zeros((3, 3) + displacement.shape[1:], dtype=np.float64)
    for component in range(3):
        gradients = np.gradient(displacement[component], axis=(0, 1, 2))
        for axis in range(3):
            jacobian[component, axis] = gradients[axis]
    jacobian[0, 0] += 1.0
    jacobian[1, 1] += 1.0
    jacobian[2, 2] += 1.0
    det = (
        jacobian[0, 0] * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
        - jacobian[0, 1] * (jacobian[1, 0] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 0])
        + jacobian[0, 2] * (jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0])
    )
    return det.astype(np.float32)


def realized_volume_change_pct(jacobian_det: np.ndarray, region_mask: np.ndarray) -> float:
    """ΔV_true = 100 * (∫_R det(J_phi) dx − V_R) / V_R (plan §3.2).

    Estimated as 100 * (mean(det) inside `region_mask` − 1); voxel volume cancels
    between the integral and V_R so no voxel-size bookkeeping is needed.
    """
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != jacobian_det.shape:
        raise ValueError(f"region_mask shape {mask.shape} != jacobian shape {jacobian_det.shape}")
    if not mask.any():
        raise ValueError("region_mask must contain at least one voxel")
    return float(100.0 * (np.mean(jacobian_det[mask].astype(np.float64)) - 1.0))


def calibrate_magnitude_for_target_change(
    field_builder: Callable[[float], DeformationField],
    region_mask: np.ndarray,
    *,
    target_change_pct: float,
    magnitude_bounds: tuple[float, float],
    max_iterations: int = 40,
    tolerance_pct: float = 0.1,
) -> tuple[float, DeformationField, float]:
    """Bisection search for the field magnitude whose *realized* volume change over
    `region_mask` matches `target_change_pct` within `tolerance_pct`.

    `field_builder` maps a single scalar (magnitude, or scale for
    `global_scaling_field`) to a `DeformationField`; realized change is assumed
    monotonic in that scalar over `magnitude_bounds` and this is checked at the
    bounds, not assumed silently. Returns (chosen_magnitude, field, realized_pct) --
    the realized value, not `target_change_pct`, is what plan §3.2 requires as the
    ground truth recorded downstream; this function only selects which field to use.
    """
    low, high = magnitude_bounds

    def realized(magnitude: float) -> float:
        return realized_volume_change_pct(jacobian_determinant(field_builder(magnitude)), region_mask)

    realized_low, realized_high = realized(low), realized(high)
    if not min(realized_low, realized_high) <= target_change_pct <= max(realized_low, realized_high):
        raise ValueError(
            f"Target change {target_change_pct}% is outside the achievable range "
            f"[{min(realized_low, realized_high):.3f}%, {max(realized_low, realized_high):.3f}%] "
            f"for magnitude_bounds={magnitude_bounds}"
        )
    increasing = realized_high >= realized_low
    magnitude = 0.5 * (low + high)
    for _ in range(max_iterations):
        magnitude = 0.5 * (low + high)
        realized_mid = realized(magnitude)
        if abs(realized_mid - target_change_pct) <= tolerance_pct:
            break
        if (realized_mid < target_change_pct) == increasing:
            low = magnitude
        else:
            high = magnitude
    field = field_builder(magnitude)
    return magnitude, field, realized(magnitude)
