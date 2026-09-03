from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import VisitFeatureArtifact, sha256_file, write_json_atomic


def compare_feature_artifacts(
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, object]:
    baseline_file = Path(baseline_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    baseline = VisitFeatureArtifact.load(baseline_file)
    candidate = VisitFeatureArtifact.load(candidate_file)
    if baseline.feature_names != candidate.feature_names:
        raise ValueError("Feature schemas differ")
    identity_arrays = (
        ("subject IDs", baseline.subject_ids, candidate.subject_ids),
        ("visit IDs", baseline.visit_ids, candidate.visit_ids),
        ("elapsed times", baseline.elapsed_years, candidate.elapsed_years),
    )
    for label, first, second in identity_arrays:
        if not np.array_equal(first, second):
            raise ValueError(f"Artifact {label} differ")
    first = baseline.X.astype(np.float64)
    second = candidate.X.astype(np.float64)
    difference = second - first
    row_denominators = np.maximum(np.linalg.norm(first, axis=1), 1e-12)
    row_relative_l2 = np.linalg.norm(difference, axis=1) / row_denominators
    symmetric_change = 2 * np.abs(difference) / (np.abs(first) + np.abs(second) + 1e-8)
    per_visit = [
        {
            "subject_id": str(subject),
            "visit_id": str(visit),
            "relative_l2": float(value),
        }
        for subject, visit, value in zip(
            baseline.subject_ids, baseline.visit_ids, row_relative_l2
        )
    ]
    flattened_correlation = (
        float(np.corrcoef(first.ravel(), second.ravel())[0, 1])
        if first.size > 1
        else 1.0
    )
    return {
        "baseline_path": str(baseline_file),
        "baseline_sha256": sha256_file(baseline_file),
        "candidate_path": str(candidate_file),
        "candidate_sha256": sha256_file(candidate_file),
        "n_visits": int(first.shape[0]),
        "n_features": int(first.shape[1]),
        "schema_equal": True,
        "row_identity_equal": True,
        "overall_relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(first), 1e-12)
        ),
        "per_visit_relative_l2_min": float(row_relative_l2.min()),
        "per_visit_relative_l2_mean": float(row_relative_l2.mean()),
        "per_visit_relative_l2_max": float(row_relative_l2.max()),
        "flattened_pearson": flattened_correlation,
        "symmetric_relative_change_median": float(np.median(symmetric_change)),
        "symmetric_relative_change_p95": float(np.quantile(symmetric_change, 0.95)),
        "per_visit": per_visit,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = compare_feature_artifacts(args.baseline, args.candidate)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
