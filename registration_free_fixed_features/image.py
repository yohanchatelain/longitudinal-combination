from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from .config import ArchitectureConfig


class ImageQCError(ValueError):
    """Raised when an image or mask violates the locked input contract."""


@dataclass(frozen=True)
class QCRecord:
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]
    mask_voxels: int
    mask_volume_mm3: float
    largest_component_fraction: float
    boundary_fraction: float
    left_fraction: float
    right_fraction: float
    physical_fov_mm: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedImage:
    image: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    qc: QCRecord
    corrected_image_sha256: str


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"|float32-le|")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_3d(path: str | Path) -> nib.spatialimages.SpatialImage:
    loaded = nib.load(str(path))
    if len(loaded.shape) > 3:
        loaded = nib.squeeze_image(loaded)
    canonical = nib.as_closest_canonical(loaded, enforce_diag=False)
    if len(canonical.shape) != 3:
        raise ImageQCError(f"Expected a 3D image, got shape {canonical.shape} at {path}")
    return canonical


def _spacing_and_shear(affine: np.ndarray) -> tuple[np.ndarray, float]:
    basis = np.asarray(affine[:3, :3], dtype=float)
    spacing = np.linalg.norm(basis, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ImageQCError(f"Invalid voxel spacing derived from affine: {spacing}")
    directions = basis / spacing
    gram = directions.T @ directions
    off_diagonal = gram - np.eye(3)
    max_shear = float(np.max(np.abs(off_diagonal)))
    return spacing, max_shear


def _boundary_fraction(mask: np.ndarray) -> float:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    return float(np.count_nonzero(mask & boundary) / max(1, np.count_nonzero(mask)))


def validate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    config: ArchitectureConfig,
) -> QCRecord:
    if image.ndim != 3 or mask.ndim != 3 or image.shape != mask.shape:
        raise ImageQCError(f"Image/mask shape mismatch: {image.shape} versus {mask.shape}")
    if not np.isfinite(image).all():
        raise ImageQCError("Image contains NaN or infinite values")
    spacing, max_shear = _spacing_and_shear(affine)
    qc = config.qc
    if np.any(spacing < qc.min_spacing_mm) or np.any(spacing > qc.max_spacing_mm):
        raise ImageQCError(
            f"Voxel spacing {spacing.tolist()} outside supported range "
            f"[{qc.min_spacing_mm}, {qc.max_spacing_mm}] mm"
        )
    if max_shear > qc.max_shear_cosine:
        raise ImageQCError(
            f"Affine shear cosine {max_shear:.6g} exceeds {qc.max_shear_cosine:.6g}"
        )
    mask = np.asarray(mask, dtype=bool)
    n_mask = int(mask.sum())
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    mask_volume = n_mask * voxel_volume
    if n_mask < qc.min_mask_voxels:
        raise ImageQCError(f"Mask has {n_mask} voxels; minimum is {qc.min_mask_voxels}")
    if mask_volume < qc.min_mask_volume_mm3:
        raise ImageQCError(
            f"Mask volume {mask_volume:.1f} mm3; minimum is {qc.min_mask_volume_mm3:.1f}"
        )
    labels, count = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=int)
    largest_fraction = float(component_sizes.max() / n_mask) if component_sizes.size else 0.0
    if largest_fraction < qc.min_largest_component_fraction:
        raise ImageQCError(
            f"Largest mask component fraction {largest_fraction:.4f} is below "
            f"{qc.min_largest_component_fraction:.4f}"
        )
    boundary_fraction = _boundary_fraction(mask)
    if boundary_fraction > qc.max_boundary_fraction:
        raise ImageQCError(
            f"Mask boundary fraction {boundary_fraction:.4f} exceeds "
            f"{qc.max_boundary_fraction:.4f}; possible truncation"
        )
    mask_indices = np.argwhere(mask)
    homogeneous = np.column_stack([mask_indices, np.ones(len(mask_indices), dtype=float)])
    world_x = (affine @ homogeneous.T).T[:, 0]
    center_x = float(np.median(world_x))
    left_fraction = float(np.mean(world_x < center_x))
    right_fraction = float(np.mean(world_x > center_x))
    if min(left_fraction, right_fraction) < qc.min_side_fraction:
        raise ImageQCError(
            f"Left/right coverage fractions ({left_fraction:.3f}, {right_fraction:.3f}) "
            f"fall below {qc.min_side_fraction:.3f}"
        )
    orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
    return QCRecord(
        shape=tuple(int(value) for value in image.shape),
        spacing_mm=tuple(float(value) for value in spacing),
        orientation=orientation,
        mask_voxels=n_mask,
        mask_volume_mm3=mask_volume,
        largest_component_fraction=largest_fraction,
        boundary_fraction=boundary_fraction,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
        physical_fov_mm=tuple(float(n * s) for n, s in zip(image.shape, spacing)),
    )


def _n4_correct(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    iterations: tuple[int, ...],
) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "simpleitk_n4 was requested but SimpleITK is unavailable. "
            "Install SimpleITK or use development-only bias_backend=none."
        ) from exc
    thread_count = int(os.environ.get("RFFF_N4_THREADS", "1"))
    if thread_count < 1:
        raise ValueError("RFFF_N4_THREADS must be a positive integer")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(thread_count)
    # SimpleITK arrays are z, y, x; the public spacing remains x, y, z.
    sitk_image = sitk.GetImageFromArray(np.transpose(image, (2, 1, 0)).astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)).astype(np.uint8))
    sitk_image.SetSpacing(tuple(float(value) for value in spacing))
    sitk_mask.CopyInformation(sitk_image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(value) for value in iterations])
    corrected = corrector.Execute(sitk_image, sitk_mask)
    return np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0)).astype(np.float32)


def robust_normalize(
    image: np.ndarray,
    mask: np.ndarray,
    clip_percentiles: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(image[mask], dtype=np.float64)
    if values.size == 0:
        raise ImageQCError("Cannot normalize an empty mask")
    low, high = np.percentile(values, clip_percentiles)
    clipped = np.clip(image, low, high).astype(np.float32, copy=False)
    clipped_values = clipped[mask]
    median = float(np.median(clipped_values))
    q25, q75 = np.percentile(clipped_values, [25.0, 75.0])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        raise ImageQCError("Masked image has zero or nearly zero robust intensity scale")
    normalized = np.zeros_like(clipped, dtype=np.float32)
    normalized[mask] = (clipped_values - median) / scale
    return normalized


def load_and_prepare(
    image_path: str | Path,
    mask_path: str | Path,
    config: ArchitectureConfig,
) -> PreparedImage:
    image_obj = _canonical_3d(image_path)
    mask_obj = _canonical_3d(mask_path)
    image = np.asarray(image_obj.dataobj, dtype=np.float32)
    mask = np.asarray(mask_obj.dataobj) > 0
    if image.shape != mask.shape or not np.allclose(image_obj.affine, mask_obj.affine, atol=1e-4):
        raise ImageQCError("Canonical image and mask do not share the same grid and affine")
    pre_qc = validate_image_and_mask(image, mask, image_obj.affine, config)
    if config.preprocessing.bias_backend == "simpleitk_n4":
        image = _n4_correct(
            image,
            mask,
            pre_qc.spacing_mm,
            config.preprocessing.n4_iterations,
        )
    corrected_image_sha256 = _array_sha256(image)
    normalized = robust_normalize(image, mask, config.preprocessing.clip_percentiles)
    final_qc = validate_image_and_mask(normalized, mask, image_obj.affine, config)
    return PreparedImage(
        image=normalized,
        mask=mask.astype(bool, copy=False),
        affine=np.asarray(image_obj.affine, dtype=float),
        qc=final_qc,
        corrected_image_sha256=corrected_image_sha256,
    )
