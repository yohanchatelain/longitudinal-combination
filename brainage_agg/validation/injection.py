"""Deterministic subject assignment and raw-volume spatial injections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def deterministic_subject_assignment(
    subject_ids: Iterable[str],
    *,
    n_validation: int,
    n_pilot: int,
    seed: int,
) -> pd.DataFrame:
    """Assign disjoint pilot/validation subjects and balanced synthetic groups.

    Input order never affects the assignment. Duplicate IDs are collapsed and
    an error is raised rather than silently reducing a requested sample.
    """
    ids = sorted({str(subject_id) for subject_id in subject_ids})
    requested = int(n_validation) + int(n_pilot)
    if n_validation < 2 or n_pilot < 2:
        raise ValueError("Pilot and validation samples must each contain at least two subjects")
    if len(ids) < requested:
        raise ValueError(f"Requested {requested} subjects but only {len(ids)} are available")

    rng = np.random.default_rng(seed)
    selected = np.asarray(ids, dtype=object)[rng.permutation(len(ids))[:requested]]
    rows: list[dict[str, object]] = []
    start = 0
    for split, count in (("pilot", n_pilot), ("validation", n_validation)):
        split_ids = selected[start : start + count]
        start += count
        # A second deterministic shuffle prevents group membership from being
        # tied to the ordering of the first sample draw.
        split_ids = split_ids[rng.permutation(len(split_ids))]
        for index, subject_id in enumerate(split_ids):
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "split": split,
                    "synthetic_group": "injected" if index % 2 == 0 else "control",
                    "assignment_seed": int(seed),
                }
            )
    result = pd.DataFrame(rows).sort_values(["split", "subject_id"]).reset_index(drop=True)
    for split, frame in result.groupby("split"):
        counts = frame["synthetic_group"].value_counts()
        if abs(int(counts.get("injected", 0)) - int(counts.get("control", 0))) > 1:
            raise AssertionError(f"Unbalanced deterministic assignment for {split}")
    return result


def tapered_label_mask(
    atlas: np.ndarray,
    label_ids: Iterable[int],
    boundary_voxels: int = 2,
) -> np.ndarray:
    """Return a [0, 1] injection field with a tapered *internal* boundary.

    The field is exactly zero outside the requested labels. With a two-voxel
    taper the first and second internal shells receive 1/3 and 2/3 strength;
    voxels at least three voxels from the edge receive full strength.
    """
    atlas = np.asarray(atlas)
    labels = tuple(int(label) for label in label_ids)
    if atlas.ndim != 3:
        raise ValueError(f"Expected a 3-D atlas, got shape {atlas.shape}")
    if boundary_voxels < 0:
        raise ValueError("boundary_voxels must be non-negative")
    binary = np.isin(atlas, labels)
    if not binary.any():
        raise ValueError(f"None of the requested labels {labels} occur in the atlas")
    if boundary_voxels == 0:
        return binary.astype(np.float32)
    distance = distance_transform_edt(binary)
    tapered = np.minimum(distance / float(boundary_voxels + 1), 1.0)
    tapered[~binary] = 0.0
    return tapered.astype(np.float32)


def apply_multiplicative_attenuation(
    volume: np.ndarray,
    injection_field: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Attenuate a raw T1 volume before any scaling/channel construction."""
    volume = np.asarray(volume, dtype=np.float32)
    field = np.asarray(injection_field, dtype=np.float32)
    if volume.shape != field.shape:
        raise ValueError(f"Volume/mask shape mismatch: {volume.shape} != {field.shape}")
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("amplitude must be in [0, 1)")
    if np.nanmin(field) < 0.0 or np.nanmax(field) > 1.0:
        raise ValueError("injection_field must be bounded by [0, 1]")
    return (volume * (1.0 - float(amplitude) * field)).astype(np.float32)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Absolute two-sample Cohen's d using the pooled sample variance."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Cohen's d requires at least two finite values per group")
    pooled_num = (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    pooled_den = len(a) + len(b) - 2
    pooled_sd = np.sqrt(pooled_num / pooled_den)
    if pooled_sd == 0:
        return float("inf") if not np.isclose(a.mean(), b.mean()) else 0.0
    return float(abs(a.mean() - b.mean()) / pooled_sd)


def select_calibrated_amplitude(
    control_values: np.ndarray,
    injected_values: Mapping[float, np.ndarray],
    *,
    target_d: float = 0.8,
) -> tuple[float, pd.DataFrame]:
    """Choose the preregistered amplitude nearest target d using pilot values.

    Ties are resolved toward the smaller attenuation. The caller is expected
    to supply values from the disjoint pilot split and must not use attribution
    maps in this calibration.
    """
    if not injected_values:
        raise ValueError("At least one candidate amplitude is required")
    rows = []
    for amplitude in sorted(float(a) for a in injected_values):
        effect = cohens_d(np.asarray(injected_values[amplitude]), control_values)
        rows.append({"amplitude": amplitude, "cohens_d": effect, "distance_to_target": abs(effect - target_d)})
    table = pd.DataFrame(rows).sort_values(["distance_to_target", "amplitude"]).reset_index(drop=True)
    return float(table.iloc[0]["amplitude"]), table.sort_values("amplitude").reset_index(drop=True)
