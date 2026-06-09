from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Empty brain mask.")
    return mask


def _masked_minmax(ch: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = _validate_mask(mask)
    ch = np.asarray(ch, dtype=np.float32)
    mn = float(np.min(ch[mask]))
    mx = float(np.max(ch[mask]))
    out = np.zeros_like(ch, dtype=np.float32)
    if mx - mn >= eps:
        out[mask] = (ch[mask] - mn) / (mx - mn + eps)
    return out


def scale_zscore(img, mask, clip_low=1, clip_high=99, eps=1e-8):
    """Clip using brain-mask percentiles, then z-score inside mask."""
    mask = _validate_mask(mask)
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    brain = clipped[mask]
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (brain - float(brain.mean())) / (float(brain.std()) + eps)
    return out


def scale_robust(img, mask, clip_low=1, clip_high=99, eps=1e-8):
    """Clip using brain-mask percentiles, then median/IQR scale inside mask."""
    mask = _validate_mask(mask)
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    brain = clipped[mask]
    q25, q75 = np.percentile(brain, [25, 75])
    iqr = float(q75 - q25)
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (brain - float(np.median(brain))) / (iqr + eps)
    return out


def scale_minmax(img, mask, clip_low=1, clip_high=99, eps=1e-8):
    """Clip using brain-mask percentiles, then min-max scale to [0, 1]."""
    mask = _validate_mask(mask)
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (clipped[mask] - lo) / (float(hi - lo) + eps)
    return out


def sobel_magnitude(img, mask):
    """Compute masked, min-max normalized 3D Sobel magnitude."""
    img = np.asarray(img, dtype=np.float32)
    sx = ndi.sobel(img, axis=0)
    sy = ndi.sobel(img, axis=1)
    sz = ndi.sobel(img, axis=2)
    mag = np.sqrt(sx * sx + sy * sy + sz * sz).astype(np.float32)
    return _masked_minmax(mag, mask)


def local_rank_channel(img, mask, size=3):
    """Local percentile rank: fraction of neighborhood values <= center voxel."""
    img = np.asarray(img, dtype=np.float32)
    mask = _validate_mask(mask)

    def rank_func(values):
        center = values[len(values) // 2]
        return float(np.mean(values <= center))

    ranked = ndi.generic_filter(
        img,
        function=rank_func,
        footprint=np.ones((size, size, size), dtype=bool),
        mode="nearest",
    ).astype(np.float32)
    ranked[~mask] = 0.0
    return ranked


def median_channel(img, mask, size=3):
    median = ndi.percentile_filter(np.asarray(img, dtype=np.float32), percentile=50, size=size)
    return _masked_minmax(median, mask)


def build_channels(
    img,
    mask,
    scaler: str,
    channels: list[str],
    clip_low=1,
    clip_high=99,
    rank_size=3,
) -> np.ndarray:
    scalers = {
        "zscore": scale_zscore,
        "robust": scale_robust,
        "minmax": scale_minmax,
    }
    if scaler not in scalers:
        raise ValueError(f"Unsupported scaler: {scaler}")

    base = scalers[scaler](img, mask, clip_low=clip_low, clip_high=clip_high)
    built = []
    for channel in channels:
        if channel == "t1":
            built.append(base)
        elif channel == "sobel":
            built.append(sobel_magnitude(base, mask))
        elif channel == "rank":
            built.append(local_rank_channel(base, mask, size=rank_size))
        elif channel == "median":
            built.append(median_channel(base, mask, size=rank_size))
        else:
            raise ValueError(f"Unsupported channel: {channel}")
    return np.stack(built, axis=0).astype(np.float32)
