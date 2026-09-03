"""Adapter wiring a CNN backbone into the abstract `response_fn` shape used
throughout this package.

Neither `self_calibration.compute_self_calibration` nor
`invariance_layer.calibrate_filter_cutoff` has any notion of what a "response" is --
both take an arbitrary `response_fn(baseline, followup) -> float` so they stay
decoupled from the model. This module defines the concrete response used across the
geometric-validation experiments: the L2 norm of the change in a backbone's feature
vector between a baseline and its follow-up, for any combination of
{raw, invariant} preprocessing variant x {frozen-random, trained} backbone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from brainage_agg.experiment.run_voxel_importance import _build_cnn as build_frozen_cnn

from .invariance_layer import preprocess_pair_with_variant

__all__ = [
    "build_frozen_cnn",
    "build_response_fn",
    "extract_features",
    "feature_change_magnitude",
]


def extract_features(
    model: torch.nn.Module,
    channels: np.ndarray,
    *,
    device: str = "cpu",
    downsample_factor: int = 1,
) -> np.ndarray:
    """One forward pass through a frozen/trained backbone.

    Same pattern as `run_attribution_validation._model_features`, reused here since
    both the frozen (`build_frozen_cnn`, re-exported from
    `run_voxel_importance._build_cnn`) and trained
    (`trained_cnn.load_trained_checkpoint`) arms return the same bare
    `CNN3D_DoubleConv`/`CNN3D_CovPool` module type.
    """
    tensor = torch.as_tensor(channels[None], dtype=torch.float32, device=device)
    if downsample_factor > 1:
        tensor = F.interpolate(tensor, scale_factor=1.0 / downsample_factor, mode="trilinear", align_corners=False)
    with torch.no_grad():
        features = model(tensor).detach().cpu().numpy()[0]
    return features.astype(np.float32)


def feature_change_magnitude(baseline_features: np.ndarray, followup_features: np.ndarray) -> float:
    """L2 norm of the feature-vector change -- the response used everywhere in this
    package (self-calibration null draws, invariance-layer filter calibration, ...).
    """
    if baseline_features.shape != followup_features.shape:
        raise ValueError(
            f"baseline/follow-up feature shape mismatch: "
            f"{baseline_features.shape} != {followup_features.shape}"
        )
    return float(np.linalg.norm(followup_features.astype(np.float64) - baseline_features.astype(np.float64)))


def build_response_fn(
    model: torch.nn.Module,
    mask: np.ndarray,
    *,
    variant: str,
    sigma_voxels: float,
    scaler: str,
    channels: list[str],
    clip_low: float = 1,
    clip_high: float = 99,
    rank_size: int = 3,
    downsample_factor: int = 1,
    device: str = "cpu",
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build a `response_fn(baseline_raw, followup_raw) -> float` for one CNN variant.

    Composes `preprocess_pair_with_variant` (raw/invariant preprocessing through a
    shared baseline/follow-up call site) with `extract_features` and
    `feature_change_magnitude` -- exactly the shape
    `self_calibration.compute_self_calibration` and
    `invariance_layer.calibrate_filter_cutoff`'s response arrays expect. The caller
    supplies whichever backbone it wants tested (frozen-random via
    `build_frozen_cnn`, or trained via `trained_cnn.load_trained_checkpoint`), so
    this function has no opinion on frozen vs. trained -- only on how "response" is
    computed once a backbone is given.
    """
    def response_fn(baseline_raw: np.ndarray, followup_raw: np.ndarray) -> float:
        baseline_channels, followup_channels = preprocess_pair_with_variant(
            baseline_raw, followup_raw, mask,
            variant=variant, sigma_voxels=sigma_voxels, scaler=scaler, channels=channels,
            clip_low=clip_low, clip_high=clip_high, rank_size=rank_size,
        )
        baseline_features = extract_features(
            model, baseline_channels, device=device, downsample_factor=downsample_factor,
        )
        followup_features = extract_features(
            model, followup_channels, device=device, downsample_factor=downsample_factor,
        )
        return feature_change_magnitude(baseline_features, followup_features)

    return response_fn
