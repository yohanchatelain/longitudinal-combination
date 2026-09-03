"""Apply a deformation field to a baseline image to build its synthetic follow-up.

Plan §3.1: "A smooth, invertible deformation field is applied only to the synthetic
follow-up. Intensities move with the anatomy; intensity is not simply attenuated
inside the target ROI." The forward map phi(x) = x + d(x) (`deformation.py`)
describes where baseline anatomy at x moves to in the follow-up; to build the
follow-up image on the same grid we need, for every output voxel y, the baseline
intensity at phi^-1(y) -- computed here by numerically inverting the displacement
field, since only the rigid and global-scaling fields have a closed-form inverse.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .deformation import DeformationField, _index_grid
from .determinism import assert_float32


def _sample_vector_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (3, D, H, W) vector field at arbitrary coordinates.

    Uses edge-value extrapolation (`mode="nearest"`), not zero-padding: zero-padding
    a displacement field at the volume boundary breaks the fixed-point iteration in
    `invert_displacement_field` for any point whose sample path leaves the grid
    (verified: with zero-padding, corner-voxel inversion error was >2 voxels; with
    edge extrapolation it is ~1e-6). Real anatomical targets sit well inside the
    brain mask, but the inversion should not be silently wrong at the grid edges.
    """
    return np.stack(
        [ndi.map_coordinates(field[component], points, order=1, mode="nearest")
         for component in range(field.shape[0])]
    ).astype(np.float32)


def invert_displacement_field(displacement: np.ndarray, *, iterations: int = 12) -> np.ndarray:
    """Numerically invert a displacement field via fixed-point iteration.

    Starting from d_inv_0 = -d, repeatedly refine d_inv(y) = -d(y + d_inv(y)) -- the
    same iteration used by e.g. ITK's InvertDisplacementFieldImageFilter for smooth,
    diffeomorphic fields. Converges when the forward map is genuinely invertible;
    `qc.py`'s positive-Jacobian check is the precondition a field should already have
    satisfied before it reaches this function.
    """
    grid = _index_grid(displacement.shape[1:])
    inverse = -displacement.astype(np.float32)
    for _ in range(iterations):
        sample_points = grid + inverse
        inverse = -_sample_vector_field(displacement, sample_points)
    return inverse.astype(np.float32)


def _resample(volume: np.ndarray, inverse_displacement: np.ndarray, *, order: int, mode: str) -> np.ndarray:
    grid = _index_grid(volume.shape)
    sample_points = grid + inverse_displacement
    return ndi.map_coordinates(volume, sample_points, order=order, mode=mode).astype(np.float32)


def warp_image(
    image: np.ndarray,
    field: DeformationField,
    *,
    order: int = 1,
    inversion_iterations: int = 12,
) -> np.ndarray:
    """Resample `image` (the baseline) onto the follow-up grid defined by `field`.

    Output voxel y receives baseline intensity at phi^-1(y), so anatomy genuinely
    moves with the deformation rather than intensity being attenuated in place.
    """
    assert_float32(image)
    if image.shape != field.displacement.shape[1:]:
        raise ValueError(f"image shape {image.shape} != field shape {field.displacement.shape[1:]}")
    inverse = invert_displacement_field(field.displacement, iterations=inversion_iterations)
    return _resample(image, inverse, order=order, mode="nearest")


def synthesize_followup(
    baseline_image: np.ndarray,
    baseline_mask: np.ndarray,
    field: DeformationField,
    *,
    order: int = 1,
    inversion_iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a grid-identical synthetic follow-up (image, mask) pair from a baseline.

    Image and mask are warped from the same inverse displacement (computed once),
    so they stay spatially consistent. The mask always uses nearest-neighbor
    interpolation regardless of `order`, since a linearly-interpolated boolean mask
    is not itself boolean.
    """
    assert_float32(baseline_image)
    if baseline_image.shape != field.displacement.shape[1:]:
        raise ValueError(
            f"baseline_image shape {baseline_image.shape} != field shape {field.displacement.shape[1:]}"
        )
    if baseline_mask.shape != baseline_image.shape:
        raise ValueError(f"baseline_mask shape {baseline_mask.shape} != image shape {baseline_image.shape}")
    inverse = invert_displacement_field(field.displacement, iterations=inversion_iterations)
    followup_image = _resample(baseline_image, inverse, order=order, mode="nearest")
    followup_mask = _resample(baseline_mask.astype(np.float32), inverse, order=0, mode="nearest") > 0.5
    return followup_image, followup_mask
