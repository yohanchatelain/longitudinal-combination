from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from registration_free_fixed_features.artifacts import VisitFeatureArtifact
from registration_free_fixed_features.cli import main


def test_extract_cli_writes_hashed_visit_artifact(tmp_path):
    grid = np.indices((17, 17, 17), dtype=float)
    radius = np.sqrt(np.sum((grid - 8.0) ** 2, axis=0))
    mask = radius <= 6.0
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask] = np.exp(-radius[mask] ** 2 / 12.0)
    affine = np.eye(4)
    image_path = tmp_path / "raw.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "sub-1",
                "visit_id": "visit-1",
                "elapsed_years": 0.0,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        ]
    ).to_csv(manifest_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {"bias_backend": "none"},
                "qc": {
                    "min_mask_voxels": 100,
                    "min_mask_volume_mm3": 100,
                    "max_boundary_fraction": 0.01,
                },
                "scattering": {
                    "scales_mm": [2.0],
                    "angular_orders": [0],
                    "max_order": 1,
                    "radial_shells": 1,
                    "summary_stats": ["mean"],
                    "kernel_truncate_sigma": 2.0,
                    "boundary_erosion_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "features.npz"

    main(
        [
            "extract",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
    )

    artifact = VisitFeatureArtifact.load(output_path)
    assert artifact.X.shape[0] == 1
    assert artifact.subject_ids.tolist() == ["sub-1"]
    assert artifact.metadata["config_hash"]
    assert output_path.with_suffix(".npz.json").is_file()
