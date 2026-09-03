"""Per-subject self-calibration: z-score a measurement against its own null draws.

Architecture §2.3: group-level calibration (Experiment 1) says the *method* is
calibrated on average across all pilot/validation subjects; it says nothing about
whether any one subject's measurement is trustworthy. This module makes invariance a
property of each measurement instead: for a subject, independent null resamples are
drawn from that subject's OWN baseline scan only (never the follow-up, and blind to
the subject's assigned condition/amplitude), and the subject's observed response is
normalized against the resulting null mean/SD before it enters any group-level
statistic (paired test, LME, bootstrap).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .deformation import DeformationField, rigid_subvoxel_field
from .synthesis import synthesize_followup


@dataclass(frozen=True)
class SelfCalibrationResult:
    subject_id: str
    observed_response: float
    null_mean: float
    null_std: float
    null_responses: np.ndarray
    z_score: float


def null_resample_fields(
    shape: tuple[int, int, int],
    *,
    n_resamples: int,
    seed: int,
    max_translation_voxels: float = 0.5,
    max_rotation_degrees: float = 1.0,
) -> list[DeformationField]:
    """Independent resampling-sham fields for one subject's null distribution.

    Each draw is a `rigid_subvoxel_field` -- true anatomical change exactly zero by
    construction (plan §3.1) -- with an independently sampled small
    translation/rotation, sized to a plausible registration-jitter regime rather than
    gross motion. This probes interpolation/registration sensitivity with no
    dependence on the subject's actual assigned condition or amplitude.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(n_resamples):
        translation = rng.uniform(-max_translation_voxels, max_translation_voxels, size=3)
        rotation = rng.uniform(-max_rotation_degrees, max_rotation_degrees, size=3)
        fields.append(
            rigid_subvoxel_field(
                shape,
                translation_voxels=tuple(translation),
                rotation_degrees=tuple(rotation),
            )
        )
    return fields


def compute_null_response_distribution(
    baseline_image: np.ndarray,
    baseline_mask: np.ndarray,
    response_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int,
    seed: int,
    max_translation_voxels: float = 0.5,
    max_rotation_degrees: float = 1.0,
) -> np.ndarray:
    """This subject's null response distribution, from `n_resamples` independent
    resampling-sham draws of their own baseline scan.

    Split out from `compute_self_calibration` so a caller scoring *several*
    responses for the same subject (e.g. one per synthetic condition) computes this
    null distribution once and z-scores every condition against the same shared
    null, rather than resynthesizing an independent (if reproducible) null per
    condition and z-scoring each against a different draw.
    """
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2 to estimate a null standard deviation")
    fields = null_resample_fields(
        baseline_image.shape, n_resamples=n_resamples, seed=seed,
        max_translation_voxels=max_translation_voxels, max_rotation_degrees=max_rotation_degrees,
    )
    return np.array(
        [
            float(response_fn(baseline_image, synthesize_followup(baseline_image, baseline_mask, field)[0]))
            for field in fields
        ],
        dtype=float,
    )


def zscore_against_null(observed_response: float, null_responses: np.ndarray, *, min_null_std: float = 1e-6) -> dict:
    """Z-score one observed response against an already-computed null distribution.

    `min_null_std` floors the denominator so a degenerate (zero-variance) null
    distribution cannot produce an infinite z-score.
    """
    null_responses = np.asarray(null_responses, dtype=float)
    if len(null_responses) < 2:
        raise ValueError("null_responses must contain at least two values")
    null_mean = float(np.mean(null_responses))
    null_std = float(np.std(null_responses, ddof=1))
    z_score = (float(observed_response) - null_mean) / max(null_std, min_null_std)
    return {"null_mean": null_mean, "null_std": null_std, "z_score": float(z_score)}


def compute_self_calibration(
    subject_id: str,
    *,
    observed_response: float,
    baseline_image: np.ndarray,
    baseline_mask: np.ndarray,
    response_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int,
    seed: int,
    max_translation_voxels: float = 0.5,
    max_rotation_degrees: float = 1.0,
    min_null_std: float = 1e-6,
) -> SelfCalibrationResult:
    """Z-score `observed_response` against this subject's own null distribution.

    `response_fn(baseline_image, followup_image) -> float` computes whatever scalar
    response the caller's method produces (e.g. frozen/trained-CNN feature-change
    magnitude, raw or invariant variant); this module has no dependency on what that
    response function actually is -- it only needs the same function applied to
    both the real pair and each null pair. When scoring several responses for the
    same subject, prefer `compute_null_response_distribution` +
    `zscore_against_null` directly so every response shares one null draw.
    """
    null_responses = compute_null_response_distribution(
        baseline_image, baseline_mask, response_fn,
        n_resamples=n_resamples, seed=seed,
        max_translation_voxels=max_translation_voxels, max_rotation_degrees=max_rotation_degrees,
    )
    scored = zscore_against_null(observed_response, null_responses, min_null_std=min_null_std)
    return SelfCalibrationResult(
        subject_id=subject_id,
        observed_response=float(observed_response),
        null_mean=scored["null_mean"],
        null_std=scored["null_std"],
        null_responses=null_responses,
        z_score=scored["z_score"],
    )


def to_dataframe(results: Sequence[SelfCalibrationResult]) -> pd.DataFrame:
    """Flatten a batch of per-subject self-calibration results for downstream stats."""
    return pd.DataFrame(
        [
            {
                "subject_id": result.subject_id,
                "observed_response": result.observed_response,
                "null_mean": result.null_mean,
                "null_std": result.null_std,
                "z_score": result.z_score,
                "n_null_resamples": len(result.null_responses),
            }
            for result in results
        ]
    )
