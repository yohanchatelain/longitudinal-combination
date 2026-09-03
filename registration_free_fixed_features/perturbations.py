from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PerturbedImage:
    image: np.ndarray
    mask: np.ndarray


def _rotation_matrix_xyz(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (math.radians(value) for value in degrees_xyz)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rigid(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    rotation_degrees_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_mm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PerturbedImage:
    rotation = _rotation_matrix_xyz(rotation_degrees_xyz)
    spacing = np.diag(np.asarray(spacing_mm, dtype=float))
    output_to_input = np.linalg.inv(spacing) @ rotation.T @ spacing
    center = (np.asarray(image.shape, dtype=float) - 1.0) / 2.0
    translation_voxels = np.asarray(translation_mm_xyz, dtype=float) / np.asarray(spacing_mm)
    offset = center - output_to_input @ (center + translation_voxels)
    transformed_image = ndimage.affine_transform(
        image,
        output_to_input,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    transformed_mask = ndimage.affine_transform(
        mask.astype(np.uint8),
        output_to_input,
        offset=offset,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    if any(
        (
            transformed_mask[0].any(),
            transformed_mask[-1].any(),
            transformed_mask[:, 0].any(),
            transformed_mask[:, -1].any(),
            transformed_mask[:, :, 0].any(),
            transformed_mask[:, :, -1].any(),
        )
    ):
        raise ValueError("Rigid perturbation cropped the brain mask")
    transformed_image[~transformed_mask] = 0
    return PerturbedImage(transformed_image, transformed_mask)


def add_noise(image: np.ndarray, mask: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.asarray(image, dtype=np.float32).copy()
    output[mask] += rng.normal(0.0, sigma, size=int(mask.sum())).astype(np.float32)
    output[~mask] = 0
    return output


def multiplicative_bias(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float,
    axis: int = 0,
) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, image.shape[axis], dtype=np.float32)
    shape = [1, 1, 1]
    shape[axis] = image.shape[axis]
    field = np.exp(strength * coordinate.reshape(shape)).astype(np.float32)
    output = np.asarray(image, dtype=np.float32) * field
    output[~mask] = 0
    return output


def monotone_intensity(image: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    output = np.zeros_like(image, dtype=np.float32)
    values = image[mask]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("Intensity transform requires non-constant masked values")
    scaled = np.clip((values - low) / (high - low), 0, 1)
    output[mask] = np.power(scaled, gamma) * (high - low) + low
    return output
