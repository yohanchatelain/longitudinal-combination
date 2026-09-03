"""Architecture/input-prior controls for attribution specificity."""

from __future__ import annotations

import numpy as np
import torch

from brainage_agg.analysis.voxel_attribution import compute_attribution_map


def constant_masked_input(reference: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Replace each channel's in-mask values by its in-mask mean."""
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if reference.ndim == 3:
        reference = reference[None, ...]
    if reference.ndim != 4 or reference.shape[1:] != mask.shape:
        raise ValueError("Expected reference=(channels,H,W,D) and matching mask")
    result = np.zeros_like(reference)
    for channel in range(reference.shape[0]):
        values = reference[channel][mask]
        result[channel][mask] = float(values.mean()) if values.size else 0.0
    return result


def phase_scramble_input(
    reference: np.ndarray,
    brain_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Randomize Fourier phases while preserving each channel's amplitude spectrum."""
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if reference.ndim == 3:
        reference = reference[None, ...]
    if reference.ndim != 4 or reference.shape[1:] != mask.shape:
        raise ValueError("Expected reference=(channels,H,W,D) and matching mask")
    rng = np.random.default_rng(seed)
    result = np.zeros_like(reference)
    for channel in range(reference.shape[0]):
        spectrum = np.fft.rfftn(reference[channel])
        random_phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
        scrambled = np.fft.irfftn(np.abs(spectrum) * np.exp(1j * random_phase), s=mask.shape).real
        source = reference[channel][mask]
        target = scrambled[mask]
        if target.std() > 0:
            scrambled = (scrambled - target.mean()) * (source.std() / target.std()) + source.mean()
        result[channel][mask] = scrambled[mask]
    return result.astype(np.float32)


def unit_weight_prior(
    model: torch.nn.Module,
    channels: np.ndarray,
    *,
    device: str,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribute equal positive feature weights for a supplied control input."""
    n_features = getattr(model, "out_features", None)
    if n_features is None:
        raise ValueError("Model must expose out_features")
    tensor = torch.as_tensor(channels[None, ...], dtype=torch.float32)
    return compute_attribution_map(
        model,
        tensor,
        np.ones(int(n_features), dtype=np.float32),
        device=device,
        downsample_factor=downsample_factor,
    )
