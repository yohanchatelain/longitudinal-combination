from __future__ import annotations

import numpy as np

from registration_free_fixed_features.config import config_from_dict
from registration_free_fixed_features.perturbations import rigid
from registration_free_fixed_features.scattering import PhysicalScatteringExtractor


def _extractor(backend="scipy", device="cpu"):
    config = config_from_dict(
        {
            "preprocessing": {"bias_backend": "none"},
            "scattering": {
                "scales_mm": [1.5, 3.0],
                "angular_orders": [0, 1],
                "max_order": 2,
                "radial_shells": 2,
                "summary_stats": ["mean"],
                "kernel_truncate_sigma": 2.5,
                "boundary_erosion_mm": 0.0,
                "backend": backend,
                "device": device,
            },
        }
    )
    return PhysicalScatteringExtractor(config)


def _phantom(shape=(21, 21, 21), spacing=1.0):
    axes = [
        (np.arange(length, dtype=np.float32) - (length - 1) / 2) * spacing
        for length in shape
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    mask = radius <= 7.0
    image = np.zeros(shape, dtype=np.float32)
    image[mask] = np.exp(-radius[mask] ** 2 / 20.0) + 0.25 * (x[mask] > 0)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    return image, mask, affine


def test_feature_schema_is_stable_across_native_resolutions():
    extractor = _extractor()
    coarse = _phantom((17, 17, 17), 1.0)
    fine = _phantom((33, 33, 33), 0.5)
    coarse_features = extractor.extract(image=coarse[0], mask=coarse[1], affine=coarse[2])
    fine_features = extractor.extract(image=fine[0], mask=fine[1], affine=fine[2])
    assert coarse_features.names == fine_features.names
    assert coarse_features.values.shape == fine_features.values.shape
    # Discretization changes values, but the physical-grid implementation should
    # keep the overall coefficient vector in the same numerical regime.
    scattering = np.asarray(
        [not name.startswith("covariate|") for name in coarse_features.names]
    )
    relative = np.linalg.norm(
        coarse_features.values[scattering] - fine_features.values[scattering]
    ) / max(
        np.linalg.norm(fine_features.values[scattering]), 1e-8
    )
    assert relative < 0.20


def test_non_cropping_translation_preserves_schema_and_is_stable():
    extractor = _extractor()
    image, mask, affine = _phantom((25, 25, 25), 1.0)
    original = extractor.extract(image=image, mask=mask, affine=affine)
    moved = rigid(image, mask, (1.0, 1.0, 1.0), translation_mm_xyz=(2.0, -1.0, 1.0))
    translated = extractor.extract(image=moved.image, mask=moved.mask, affine=affine)
    assert original.names == translated.names
    scattering = np.asarray([not name.startswith("covariate|") for name in original.names])
    relative = np.linalg.norm(
        original.values[scattering] - translated.values[scattering]
    ) / max(
        np.linalg.norm(original.values[scattering]), 1e-8
    )
    assert relative < 0.12


def test_quarter_turn_rotation_preserves_orientation_pooled_features():
    extractor = _extractor()
    image, mask, affine = _phantom((25, 25, 25), 1.0)
    original = extractor.extract(image=image, mask=mask, affine=affine)
    rotated = extractor.extract(
        image=np.rot90(image, axes=(0, 1)),
        mask=np.rot90(mask, axes=(0, 1)),
        affine=affine,
    )
    scattering = np.asarray([not name.startswith("covariate|") for name in original.names])
    relative = np.linalg.norm(
        original.values[scattering] - rotated.values[scattering]
    ) / max(np.linalg.norm(original.values[scattering]), 1e-8)
    assert relative < 1e-5


def test_torch_cpu_backend_matches_scipy_reference():
    image, mask, affine = _phantom((17, 17, 17), 1.0)
    scipy_features = _extractor("scipy", "cpu").extract(
        image=image, mask=mask, affine=affine
    )
    torch_extractor = _extractor("torch", "cpu")
    torch_features = torch_extractor.extract(image=image, mask=mask, affine=affine)
    assert torch_extractor.runtime["backend"] == "torch"
    assert torch_extractor.runtime["device"] == "cpu"
    assert scipy_features.names == torch_features.names
    np.testing.assert_allclose(torch_features.values, scipy_features.values, rtol=2e-4, atol=2e-5)
