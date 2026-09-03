from __future__ import annotations

import numpy as np

from registration_free_fixed_features.artifacts import VisitFeatureArtifact, sha256_file


def test_feature_artifact_round_trip_and_sidecar(tmp_path):
    artifact = VisitFeatureArtifact(
        X=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        feature_names=("first", "second"),
        subject_ids=np.array(["sub-1", "sub-1"]),
        visit_ids=np.array(["v1", "v2"]),
        elapsed_years=np.array([0.0, 1.0]),
        image_paths=np.array(["/image1", "/image2"]),
        image_hashes=np.array(["a", "b"]),
        mask_hashes=np.array(["c", "d"]),
        qc_json=np.array(["{}", "{}"]),
        metadata={"schema_version": "test"},
    )
    path = tmp_path / "features.npz"
    artifact.save(path)
    loaded = VisitFeatureArtifact.load(path)
    np.testing.assert_array_equal(loaded.X, artifact.X)
    assert loaded.feature_names == artifact.feature_names
    assert path.with_suffix(".npz.json").is_file()
    assert len(sha256_file(path)) == 64
