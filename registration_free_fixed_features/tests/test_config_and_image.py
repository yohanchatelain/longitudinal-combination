from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from registration_free_fixed_features.config import config_from_dict
from registration_free_fixed_features.image import ImageQCError, load_and_prepare


def _development_config():
    return config_from_dict(
        {
            "confirmatory": False,
            "qc": {
                "min_spacing_mm": 0.5,
                "max_spacing_mm": 3.0,
                "min_mask_voxels": 100,
                "min_mask_volume_mm3": 100.0,
                "min_largest_component_fraction": 0.95,
                "max_boundary_fraction": 0.01,
            },
            "preprocessing": {"bias_backend": "none"},
        }
    )


def test_confirmatory_config_rejects_noop_bias_correction():
    with pytest.raises(ValueError, match="requires a pinned N4"):
        config_from_dict(
            {"confirmatory": True, "preprocessing": {"bias_backend": "none"}}
        )


def test_raw_image_and_mask_are_canonicalized_and_normalized(tmp_path):
    grid = np.indices((17, 17, 17), dtype=float)
    radius = np.sqrt(np.sum((grid - 8.0) ** 2, axis=0))
    mask = radius <= 6.0
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask] = 100 + 20 * (1 - radius[mask] / 6.0)
    # LAS input exercises lossless axis flipping to RAS+.
    affine = np.diag([-1.2, 1.2, 1.2, 1.0])
    affine[0, 3] = 20.0
    image_path = tmp_path / "raw.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image[..., None], affine), image_path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)

    prepared = load_and_prepare(image_path, mask_path, _development_config())

    assert prepared.qc.orientation == ("R", "A", "S")
    assert prepared.image.dtype == np.float32
    assert np.all(prepared.image[~prepared.mask] == 0)
    assert abs(float(np.median(prepared.image[prepared.mask]))) < 1e-6
    assert len(prepared.corrected_image_sha256) == 64


def test_mask_touching_boundary_is_rejected(tmp_path):
    image = np.ones((12, 12, 12), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[:8, 2:10, 2:10] = 1
    affine = np.eye(4)
    image_path = tmp_path / "raw.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    with pytest.raises(ImageQCError, match="boundary fraction"):
        load_and_prepare(image_path, mask_path, _development_config())
