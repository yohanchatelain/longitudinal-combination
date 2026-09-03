from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisitFeature:
    elapsed_years: float
    values: np.ndarray


@dataclass(frozen=True)
class SubjectRepresentation:
    values: np.ndarray
    names: tuple[str, ...]


def _annualized_delta(first: np.ndarray, last: np.ndarray, elapsed: float) -> np.ndarray:
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Annualized change requires a positive finite interval")
    return (last - first) / elapsed


def build_subject_representations(
    visits: list[VisitFeature],
    feature_names: tuple[str, ...],
    *,
    min_elapsed_years: float,
    max_elapsed_years: float,
) -> dict[str, SubjectRepresentation]:
    if not visits:
        raise ValueError("At least one visit is required")
    ordered = sorted(visits, key=lambda visit: visit.elapsed_years)
    if any(visit.values.ndim != 1 or len(visit.values) != len(feature_names) for visit in ordered):
        raise ValueError("Every visit must match the immutable visit-level schema")
    times = np.asarray([visit.elapsed_years for visit in ordered], dtype=float)
    if not np.isfinite(times).all() or len(np.unique(times)) != len(times):
        raise ValueError("Visit times must be finite and unique")
    values = np.vstack([np.asarray(visit.values, dtype=np.float64) for visit in ordered])
    output = {
        "baseline": SubjectRepresentation(
            values=values[0].astype(np.float32),
            names=tuple(f"baseline|{name}" for name in feature_names),
        )
    }
    if len(ordered) < 2:
        return output
    elapsed = float(times[-1] - times[0])
    if elapsed < min_elapsed_years or elapsed > max_elapsed_years:
        raise ValueError(
            f"Elapsed time {elapsed:.4g} outside [{min_elapsed_years}, {max_elapsed_years}] years"
        )
    rate = _annualized_delta(values[0], values[-1], elapsed)
    if len(ordered) == 2:
        endpoint_mean = 0.5 * (values[0] + values[-1])
        output["mean_rate"] = SubjectRepresentation(
            values=np.concatenate([endpoint_mean, rate]).astype(np.float32),
            names=tuple(f"endpoint_mean|{name}" for name in feature_names)
            + tuple(f"annualized_rate|{name}" for name in feature_names),
        )
        output["rate"] = SubjectRepresentation(
            values=rate.astype(np.float32),
            names=tuple(f"annualized_rate|{name}" for name in feature_names),
        )
        return output

    centered_times = times - times[0]
    design = np.column_stack([np.ones(len(times)), centered_times])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    output["intercept_slope"] = SubjectRepresentation(
        values=np.concatenate([coefficients[0], coefficients[1]]).astype(np.float32),
        names=tuple(f"intercept_t0|{name}" for name in feature_names)
        + tuple(f"ols_slope|{name}" for name in feature_names),
    )
    return output


def build_candidate_matrices(
    X: np.ndarray,
    subject_ids: np.ndarray,
    elapsed_years: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    min_elapsed_years: float,
    max_elapsed_years: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    subjects = np.asarray(sorted(str(value) for value in np.unique(subject_ids)))
    per_subject: dict[str, dict[str, SubjectRepresentation]] = {}
    for subject in subjects:
        selected = np.asarray(subject_ids).astype(str) == subject
        visits = [
            VisitFeature(float(time), np.asarray(row, dtype=np.float32))
            for time, row in zip(elapsed_years[selected], X[selected])
        ]
        per_subject[subject] = build_subject_representations(
            visits,
            feature_names,
            min_elapsed_years=min_elapsed_years,
            max_elapsed_years=max_elapsed_years,
        )
    common = set.intersection(*(set(value) for value in per_subject.values()))
    if not common:
        raise ValueError("Subjects have no common longitudinal representation")
    matrices: dict[str, np.ndarray] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    for candidate in sorted(common):
        rows = [per_subject[subject][candidate] for subject in subjects]
        schemas[candidate] = rows[0].names
        if any(row.names != schemas[candidate] for row in rows):
            raise ValueError(f"Schema drift in longitudinal candidate {candidate}")
        matrices[candidate] = np.vstack([row.values for row in rows]).astype(np.float32)
    return subjects, matrices, schemas
