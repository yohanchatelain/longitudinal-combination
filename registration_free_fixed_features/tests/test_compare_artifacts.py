from pathlib import Path

import numpy as np
import pytest

from registration_free_fixed_features.artifacts import VisitFeatureArtifact
from registration_free_fixed_features.compare_artifacts import compare_feature_artifacts


def _artifact(values: np.ndarray, names: tuple[str, ...]) -> VisitFeatureArtifact:
    n = len(values)
    return VisitFeatureArtifact(
        X=values.astype(np.float32),
        feature_names=names,
        subject_ids=np.asarray([f"s{i}" for i in range(n)]),
        visit_ids=np.asarray(["v0"] * n),
        elapsed_years=np.zeros(n, dtype=np.float32),
        image_paths=np.asarray([f"image{i}" for i in range(n)]),
        image_hashes=np.asarray([f"ih{i}" for i in range(n)]),
        mask_hashes=np.asarray([f"mh{i}" for i in range(n)]),
        qc_json=np.asarray(["{}"] * n),
        metadata={},
    )


def test_compare_feature_artifacts_reports_relative_change(tmp_path: Path) -> None:
    baseline = _artifact(np.asarray([[1.0, 2.0], [3.0, 4.0]]), ("a", "b"))
    candidate = _artifact(np.asarray([[1.0, 2.0], [3.3, 4.4]]), ("a", "b"))
    baseline_path = tmp_path / "baseline.npz"
    candidate_path = tmp_path / "candidate.npz"
    baseline.save(baseline_path)
    candidate.save(candidate_path)

    report = compare_feature_artifacts(baseline_path, candidate_path)

    assert report["n_visits"] == 2
    assert report["n_features"] == 2
    assert report["per_visit_relative_l2_min"] == pytest.approx(0.0)
    assert report["per_visit_relative_l2_max"] == pytest.approx(0.1)


def test_compare_feature_artifacts_rejects_schema_drift(tmp_path: Path) -> None:
    baseline = _artifact(np.asarray([[1.0]]), ("a",))
    candidate = _artifact(np.asarray([[1.0]]), ("b",))
    baseline_path = tmp_path / "baseline.npz"
    candidate_path = tmp_path / "candidate.npz"
    baseline.save(baseline_path)
    candidate.save(candidate_path)

    with pytest.raises(ValueError, match="schemas differ"):
        compare_feature_artifacts(baseline_path, candidate_path)
