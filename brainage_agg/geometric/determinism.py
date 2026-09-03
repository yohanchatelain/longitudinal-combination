"""Deterministic execution and the reproducibility self-test ("gate 0").

Architecture §2.1: nuisance sensitivity that comes from *non-reproducibility* (mixed
floating-point precision, non-deterministic ops, or a baseline/follow-up preprocessing
call that drifts in its parameters) must be eliminated by construction, not modeled
statistically. This module is the foundation every other geometric-validation module is
built on: it must pass before any confirmatory cell's result is trusted.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from src.preprocessing import build_channels


def enforce_deterministic_environment(seed: int) -> None:
    """Pin every source of non-determinism this pipeline controls.

    Call once per process (or per worker, for a SLURM array task) before any
    resampling, preprocessing, or model forward pass. Torch is imported lazily so
    CPU-only geometric code (deformation/QC/synthesis) has no hard torch dependency.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assert_float32(*arrays: np.ndarray, context: str = "") -> None:
    """Reject any non-float32 array reaching the pipeline.

    Mixed precision is a nuisance source in its own right (rounding differs by op
    and by device); the geometric pipeline fixes float32 everywhere instead of
    tolerating and then statistically calibrating around it.
    """
    label = f"{context}: " if context else ""
    for index, array in enumerate(arrays):
        dtype = np.asarray(array).dtype
        if dtype != np.float32:
            raise TypeError(
                f"{label}array {index} has dtype {dtype}, expected float32 "
                "(mixed precision is not permitted in the geometric validation pipeline)"
            )


def preprocess_pair(
    baseline_raw: np.ndarray,
    followup_raw: np.ndarray,
    mask: np.ndarray,
    *,
    scaler: str,
    channels: list[str],
    clip_low: float = 1,
    clip_high: float = 99,
    rank_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a baseline/follow-up pair through one call site with shared kwargs.

    Both images go through the identical `build_channels` call with identical
    keyword arguments, so a baseline/follow-up preprocessing divergence can only
    come from the input arrays themselves (i.e. the deformation under test) -- never
    from a parameter mismatch between two independently-written calls.
    """
    assert_float32(baseline_raw, followup_raw, context="preprocess_pair input")
    kwargs = dict(
        scaler=scaler, channels=channels,
        clip_low=clip_low, clip_high=clip_high, rank_size=rank_size,
    )
    baseline_channels = build_channels(baseline_raw, mask, **kwargs)
    followup_channels = build_channels(followup_raw, mask, **kwargs)
    return baseline_channels, followup_channels


def reproducibility_self_test(
    pipeline: Callable[[], np.ndarray],
    *,
    tolerance: float,
    seed: int,
) -> dict[str, Any]:
    """Gate 0: run `pipeline` twice under an identical deterministic environment and
    require the outputs to match within `tolerance`.

    `pipeline` takes no arguments and must internally read whatever inputs it needs
    (typically a closure over one subject's exact-duplicate condition) so re-invoking
    it exercises the same code path a confirmatory cell would use. A failure here
    means the pipeline itself is non-reproducible and no downstream calibration or
    localization result can be trusted until it is fixed.
    """
    enforce_deterministic_environment(seed)
    first = np.asarray(pipeline(), dtype=np.float64)
    enforce_deterministic_environment(seed)
    second = np.asarray(pipeline(), dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Non-deterministic output shape: {first.shape} != {second.shape}")
    max_abs_diff = float(np.max(np.abs(first - second))) if first.size else 0.0
    return {
        "passed": max_abs_diff <= tolerance,
        "max_abs_diff": max_abs_diff,
        "tolerance": float(tolerance),
        "n_values": int(first.size),
    }
