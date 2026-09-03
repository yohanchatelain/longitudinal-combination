from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from registration_free_fixed_features.artifacts import sha256_file
from registration_free_fixed_features.config import config_from_dict
from registration_free_fixed_features import synthstrip_masks
from registration_free_fixed_features.synthstrip_masks import mask_overlap


def _save_mask(path: Path, mask: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), path)


def test_mask_overlap_reports_exact_metrics(tmp_path: Path) -> None:
    reference = np.zeros((5, 5, 5), dtype=bool)
    candidate = np.zeros_like(reference)
    reference[1:3, 1:3, 1:3] = True
    candidate[2:4, 1:3, 1:3] = True
    reference_path = tmp_path / "reference.nii.gz"
    candidate_path = tmp_path / "candidate.nii.gz"
    _save_mask(reference_path, reference)
    _save_mask(candidate_path, candidate)

    metrics = mask_overlap(reference_path, candidate_path)

    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["jaccard"] == pytest.approx(1 / 3)
    assert metrics["reference_voxels"] == 8
    assert metrics["synthstrip_voxels"] == 8
    assert metrics["synthstrip_to_reference_volume_ratio"] == pytest.approx(1.0)


def test_mask_overlap_rejects_grid_mismatch(tmp_path: Path) -> None:
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    first = tmp_path / "first.nii.gz"
    second = tmp_path / "second.nii.gz"
    _save_mask(first, mask)
    nib.save(nib.Nifti1Image(mask, np.diag([2.0, 1.0, 1.0, 1.0])), second)

    with pytest.raises(ValueError, match="same canonical grid"):
        mask_overlap(first, second)


def test_builder_accepts_independent_raw_only_manifest(tmp_path: Path, monkeypatch) -> None:
    grid = np.indices((17, 17, 17), dtype=float)
    brain = np.sqrt(np.sum((grid - 8.0) ** 2, axis=0)) <= 6.0
    intensities = brain * (1.0 + grid[0] / 17.0)
    image_path = tmp_path / "raw.nii.gz"
    nib.save(nib.Nifti1Image(intensities.astype(np.float32), np.eye(4)), image_path)
    source = tmp_path / "raw_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "sub-1",
                "visit_id": "v1",
                "elapsed_years": 0.0,
                "image_path": str(image_path),
                "confirmatory_eligible": True,
            }
        ]
    ).to_csv(source, index=False)
    container = tmp_path / "container.sif"
    container.write_bytes(b"frozen-test-container")

    def fake_run(**kwargs):
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)), kwargs["mask_path"])
        return ["fake-synthstrip"], 1.25

    monkeypatch.setattr(synthstrip_masks, "_run_synthstrip", fake_run)
    config = config_from_dict(
        {
            "qc": {
                "min_mask_voxels": 100,
                "min_mask_volume_mm3": 100,
                "max_boundary_fraction": 0.01,
            },
            "preprocessing": {"bias_backend": "none"},
        }
    )

    output = synthstrip_masks.build_synthstrip_manifest(
        source,
        tmp_path / "output",
        container,
        config,
        expected_container_sha256=sha256_file(container),
        gpu=False,
    )

    result = pd.read_csv(output)
    assert bool(result.loc[0, "confirmatory_eligible"])
    assert bool(result.loc[0, "mask_confirmatory_eligible"])
    assert Path(result.loc[0, "mask_path"]).is_file()
