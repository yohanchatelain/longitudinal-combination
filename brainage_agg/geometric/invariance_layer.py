"""The band-limiting invariance layer (architecture §2.2) and its pilot-only calibration.

A fixed (non-learned) anti-alias filter sits between the raw image and channel
construction, applied identically to every input, targeting the resampling-sham
failure mode directly: sub-voxel interpolation/registration jitter is high-frequency
by construction, so suppressing it before any derived channel (sobel, local rank,
...) is computed should keep those channels from amplifying it.

This produces two CNN variants under test everywhere downstream: `"raw"` (identity --
the existing, unmodified architecture, now the control arm) and `"invariant"` (the
filter applied). The filter's one free parameter, `sigma_voxels`, must be calibrated
on the pilot split only, exactly like the amplitude calibration in
`validation/injection.py::select_calibrated_amplitude` -- and the calibration must not
be allowed to win by suppressing real signal along with noise, so it is constrained
to retain a minimum fraction of the unfiltered dose response.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .determinism import assert_float32, preprocess_pair

VARIANTS = ("raw", "invariant")


def band_limit_filter(volume: np.ndarray, mask: np.ndarray, *, sigma_voxels: float) -> np.ndarray:
    """Mask-aware isotropic Gaussian low-pass filter (normalized convolution).

    A plain `gaussian_filter(volume * mask)` pulls voxels near the brain boundary
    dark, since the convolution kernel there averages in the zero background. This
    normalizes by the equally-smoothed mask instead (`smoothed(volume*mask) /
    smoothed(mask)`), so intensity is unbiased at any distance from the boundary --
    verified by the "uniform-intensity mask" test in test_geometric_validation.py,
    where a constant value inside the mask must come back exactly unchanged.
    """
    if sigma_voxels < 0:
        raise ValueError("sigma_voxels must be non-negative")
    assert_float32(volume, context="band_limit_filter")
    mask = np.asarray(mask, dtype=bool)
    if volume.shape != mask.shape:
        raise ValueError(f"volume shape {volume.shape} != mask shape {mask.shape}")
    if sigma_voxels == 0.0:
        out = volume.copy()
        out[~mask] = 0.0
        return out.astype(np.float32)

    weight = mask.astype(np.float32)
    smoothed_signal = ndi.gaussian_filter(volume.astype(np.float32) * weight, sigma=sigma_voxels)
    smoothed_weight = ndi.gaussian_filter(weight, sigma=sigma_voxels)
    out = np.zeros_like(volume, dtype=np.float32)
    valid = smoothed_weight > 1e-6
    out[valid] = smoothed_signal[valid] / smoothed_weight[valid]
    out[~mask] = 0.0
    return out


def apply_invariance_layer(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    variant: str,
    sigma_voxels: float,
) -> np.ndarray:
    """Dispatch to the identity (`"raw"`) or filtered (`"invariant"`) CNN variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "raw":
        assert_float32(volume, context="apply_invariance_layer")
        return volume.astype(np.float32)
    return band_limit_filter(volume, mask, sigma_voxels=sigma_voxels)


def preprocess_pair_with_variant(
    baseline_raw: np.ndarray,
    followup_raw: np.ndarray,
    mask: np.ndarray,
    *,
    variant: str,
    sigma_voxels: float,
    scaler: str,
    channels: list[str],
    clip_low: float = 1,
    clip_high: float = 99,
    rank_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a baseline/follow-up pair for one CNN variant.

    The invariance layer is applied to both images through the same call with the
    same `sigma_voxels`, then handed to `determinism.preprocess_pair`, which enforces
    the identical-`build_channels`-call-site discipline on top of it. A raw-vs-
    invariant difference can therefore only ever come from the filter itself, never a
    parameter drift between the baseline and follow-up call.
    """
    filtered_baseline = apply_invariance_layer(baseline_raw, mask, variant=variant, sigma_voxels=sigma_voxels)
    filtered_followup = apply_invariance_layer(followup_raw, mask, variant=variant, sigma_voxels=sigma_voxels)
    return preprocess_pair(
        filtered_baseline, filtered_followup, mask,
        scaler=scaler, channels=channels,
        clip_low=clip_low, clip_high=clip_high, rank_size=rank_size,
    )


@dataclass(frozen=True)
class FilterCalibrationResult:
    sigma_voxels: float
    table: pd.DataFrame


def calibrate_filter_cutoff(
    candidate_sigmas: Iterable[float],
    sham_response: Mapping[float, np.ndarray],
    dose_response: Mapping[float, np.ndarray],
    *,
    reference_dose_response: np.ndarray,
    min_dose_retention_fraction: float = 0.8,
) -> FilterCalibrationResult:
    """Pilot-only band-limit cutoff calibration.

    `sham_response[sigma]` / `dose_response[sigma]` are per-pilot-subject scalar
    responses (e.g. frozen-CNN feature-change magnitude) to the zero-change and
    dosed conditions with that sigma's invariance layer applied.
    `reference_dose_response` is the dose response with no filtering, used as the
    retention denominator so a candidate cannot win by suppressing real signal along
    with noise. Selects the candidate with the lowest mean sham response among those
    retaining at least `min_dose_retention_fraction` of the reference dose response.

    Must be computed on the pilot split only -- the same circularity discipline as
    `validation/injection.py::select_calibrated_amplitude`; this function does not
    enforce that itself since it has no notion of subject splits, only response
    arrays the caller must already have restricted to pilot subjects.
    """
    reference_dose_mean = float(np.mean(reference_dose_response))
    if reference_dose_mean <= 0:
        raise ValueError("reference_dose_response must have a positive mean to define retention")
    rows = []
    for sigma in sorted(float(candidate) for candidate in candidate_sigmas):
        if sigma not in sham_response or sigma not in dose_response:
            raise ValueError(f"Missing sham/dose response for candidate sigma={sigma}")
        sham_mean = float(np.mean(sham_response[sigma]))
        dose_mean = float(np.mean(dose_response[sigma]))
        retention = dose_mean / reference_dose_mean
        rows.append(
            {
                "sigma_voxels": sigma,
                "sham_response_mean": sham_mean,
                "dose_response_mean": dose_mean,
                "dose_retention_fraction": retention,
                "meets_retention_constraint": retention >= min_dose_retention_fraction,
            }
        )
    table = pd.DataFrame(rows).sort_values("sigma_voxels").reset_index(drop=True)
    eligible = table[table["meets_retention_constraint"]]
    if eligible.empty:
        raise ValueError(
            "No candidate sigma retains at least "
            f"{min_dose_retention_fraction:.0%} of the unfiltered dose response"
        )
    chosen = eligible.sort_values(["sham_response_mean", "sigma_voxels"]).iloc[0]
    return FilterCalibrationResult(sigma_voxels=float(chosen["sigma_voxels"]), table=table)
