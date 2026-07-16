"""Gradient-based voxel attribution for untrained CNN features.

Maps CNN feature importance weights (e.g. group-difference t-statistics) back
to voxel space via a single backpropagation pass, then projects the resulting
maps to brain regions via the FreeSurfer aparc+aseg atlas.

Both a signed (directional) and an absolute importance map are returned:
  - signed_map: positive voxels drive features elevated in group_a; negative
                voxels drive features elevated in group_b.
  - abs_map:    total voxel sensitivity regardless of direction.

High-level API (recommended entry point)
-----------------------------------------
    from brainage_agg.analysis.voxel_attribution import compute_voxel_importance

    result = compute_voxel_importance(
        model=model,            # frozen CNN3D_DoubleConv / CNN3D_CovPool
        brain_volume=volume,    # (H,W,D) numpy array OR pre-batched (1,C,H,W,D) tensor
        importance_weights=w,   # (n_features,) — e.g. t-statistics from group_diff
        atlas=atlas_volume,     # (H,W,D) int32 — FreeSurfer aparc+aseg labels
        lut="FS-LUT.txt",       # path to FS-LUT.txt, or pre-loaded {label_id: name} dict
    )

    result.signed_map          # (H,W,D) float32 — directional gradient map
    result.abs_map             # (H,W,D) float32 — absolute sensitivity map
    result.region_df           # pd.DataFrame ranked by abs_importance
    result.top_regions(n=10)   # convenience: top-n rows of region_df

Low-level API (for custom pipelines)
--------------------------------------
    compute_attribution_map(model, x_tensor, importance_weights, device, downsample_factor)
    project_to_atlas(signed_map, abs_map, atlas)
    load_freesurfer_lut(lut_path)

Activation-cached API (for multiple weight vectors per input)
---------------------------------------------------------------
    cache = forward_cache(model, x_tensor, device, downsample_factor)
    signed_map, abs_map = backprop_from_w(cache, w, model.out_features, retain_graph=True)
    # ... repeat backprop_from_w for additional weight vectors on the same cache,
    # last call with retain_graph=False ...
    compute_attribution_map(...) == forward_cache(...) + backprop_from_w(..., retain_graph=False)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_freesurfer_lut(lut_path: str | Path) -> dict[int, str]:
    """Parse a FreeSurfer Color LUT file → {label_id: region_name}.

    Format expected (matches FS-LUT.txt in project root):
        0   Unknown                0   0   0   0
        1   Left-Cerebral-Exterior 70 130 180   0
        ...
    Lines starting with '#' or blank lines are skipped.
    """
    lut: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(\S+)")
    with open(lut_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            m = pattern.match(line)
            if m:
                lut[int(m.group(1))] = m.group(2)
    return lut


def subject_parcellation_path(fs_root: str | Path, directory: str) -> Path:
    """Return path to the Desikan-Killiany aparc+aseg.mgz for a FreeSurfer subject.

    Both aparc+aseg.mgz and brain.mgz are in the same 256³ 1mm conformed space,
    so no registration is needed before overlaying the atlas on gradient maps.
    Used for both NKI and PPMI (harmonized on DK atlas).
    """
    return Path(fs_root) / str(directory) / "mri" / "aparc+aseg.mgz"


@dataclass
class AttributionCache:
    """Cached forward pass, ready for one or more `backprop_from_w` calls.

    Attributes:
        x_small:           Downsampled input, (1, C, h, w, d), requires_grad=True.
        features:          Model output, (1, out_features), graph retained.
        original_size:     (H, W, D) of the un-downsampled input — gradients are
                            upsampled back to this shape.
        device:            Device the cache lives on.
        downsample_factor: Downsampling factor used to build x_small.
    """

    x_small: torch.Tensor
    features: torch.Tensor
    original_size: tuple[int, int, int]
    device: torch.device
    downsample_factor: int


def forward_cache(
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
    downsample_factor: int = 2,
) -> AttributionCache:
    """Run the forward pass once, retaining the graph for repeated backprop.

    Downsamples the input (default factor 2: 256³→128³, see
    `compute_attribution_map` for the memory rationale), runs the frozen CNN,
    and returns an `AttributionCache` for `backprop_from_w`. On CUDA OOM,
    retries the entire forward pass on CPU.

    Args:
        model:              Frozen 3-D CNN (all params have requires_grad=False).
        x_tensor:           Input tensor (1, C, H, W, D), float32.
        device:             Compute device ('cpu' or 'cuda').
        downsample_factor:  Spatial downsampling applied before the forward pass.
                            Default 2 (256³→128³). Use 1 to skip.
    """
    import torch.nn.functional as F

    str_device = str(device)
    model = model.to(device).eval()

    x_full = x_tensor.to(device).detach()
    original_size = x_full.shape[2:]  # (H, W, D)
    if downsample_factor > 1:
        x_small = F.interpolate(
            x_full, scale_factor=1.0 / downsample_factor,
            mode="trilinear", align_corners=False,
        )
    else:
        x_small = x_full
    x_small = x_small.requires_grad_(True)

    try:
        features = model(x_small)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and "cuda" in str_device:
            torch.cuda.empty_cache()
            return forward_cache(model, x_tensor, "cpu", downsample_factor)
        raise

    return AttributionCache(
        x_small=x_small,
        features=features,
        original_size=tuple(int(s) for s in original_size),
        device=torch.device(device),
        downsample_factor=downsample_factor,
    )


def _raw_grad_small(
    cache: AttributionCache,
    importance_weights: np.ndarray,
    model_out_features: int | None = None,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Backpropagate one weight vector; return the raw gradient at the
    cached *downsampled* resolution, before channel reduction or upsampling.

    Shared by `backprop_from_w` (full signed/abs maps, channel-reduced and
    upsampled) and `permutation_null.cache_topk_gradients` (per-feature
    gradients for the feature-index-permutation null), which only need this
    raw `(C, h, w, d)` tensor.

    Args:
        cache:               Result of `forward_cache`.
        importance_weights:  (n_features,) signed weights. For concatenation
                             aggregations whose weight vector is k×n_features,
                             segments are averaged to n_features.
        model_out_features:  Model's `out_features` (e.g. `model.out_features`),
                             used for the k×n_feat → n_feat reconciliation above.
                             Pass `None` to skip (weight length must already match).
        retain_graph:        Pass `True` for all but the last backward pass on
                             a given `cache` (PyTorch frees the graph after a
                             `retain_graph=False` backward pass).

    Returns:
        (C, h, w, d) tensor — d(scalar)/d(cache.x_small), on `cache.device`.
    """
    raw_w = np.asarray(importance_weights, dtype=np.float32)
    if model_out_features is not None and raw_w.shape[0] != model_out_features:
        if raw_w.shape[0] % model_out_features == 0:
            n_seg = raw_w.shape[0] // model_out_features
            raw_w = raw_w.reshape(n_seg, model_out_features).mean(axis=0)
        else:
            raise ValueError(
                f"importance_weights length ({raw_w.shape[0]}) is not a multiple "
                f"of model output dim ({model_out_features}). Check arch/channels."
            )
    w = torch.tensor(raw_w, dtype=torch.float32, device=cache.device)

    if cache.x_small.grad is not None:
        cache.x_small.grad = None

    scalar = (cache.features[0] * w).sum()
    scalar.backward(retain_graph=retain_graph)

    return cache.x_small.grad[0].detach()  # (C, h, w, d)


def backprop_from_w(
    cache: AttributionCache,
    importance_weights: np.ndarray,
    model_out_features: int | None = None,
    retain_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Backpropagate one importance-weight vector through a cached forward pass.

    Computes:
        scalar = sum_i( w_i * cache.features_i )
        grad   = d(scalar) / d(cache.x_small)   (at downsampled resolution)
        → trilinearly upsampled back to cache.original_size

    Args:
        cache:               Result of `forward_cache`.
        importance_weights:  (n_features,) signed weights. For concatenation
                             aggregations whose weight vector is k×n_features,
                             segments are averaged to n_features.
        model_out_features:  Model's `out_features` (e.g. `model.out_features`),
                             used for the k×n_feat → n_feat reconciliation above.
                             Pass `None` to skip (weight length must already match).
        retain_graph:        Pass `True` for all but the last `backprop_from_w`
                             call on a given `cache` (PyTorch frees the graph
                             after a `retain_graph=False` backward pass).

    Returns:
        signed_map: (H, W, D) float32 — mean gradient over C channels, retains sign.
        abs_map:    (H, W, D) float32 — channel-wise |gradient| averaged over channels.
    """
    import torch.nn.functional as F

    grad_small = _raw_grad_small(cache, importance_weights, model_out_features, retain_graph)
    if cache.downsample_factor > 1:
        grad_full = F.interpolate(
            grad_small.unsqueeze(0), size=cache.original_size,
            mode="trilinear", align_corners=False,
        )[0]
    else:
        grad_full = grad_small

    grad_np = grad_full.cpu().numpy()                        # (C, H, W, D)
    signed_map = grad_np.mean(axis=0).astype(np.float32)    # (H, W, D)
    abs_map = np.abs(grad_np).mean(axis=0).astype(np.float32)
    return signed_map, abs_map


def compute_attribution_map(
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    importance_weights: np.ndarray,
    device: str | torch.device = "cpu",
    downsample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Backpropagate importance-weighted CNN features to input voxels.

    Backward-compatible single-weight wrapper: `forward_cache` followed by a
    single `backprop_from_w(..., retain_graph=False)` call. For multiple
    weight vectors per input (e.g. w_t/w_shap/w_hybrid), call `forward_cache`
    once and `backprop_from_w` per vector instead — this avoids redundant
    forward passes.

    Downsampling the input (default factor 2: 256³→128³) reduces peak GPU
    memory by 8x (first-layer activations: 4.3 GB→0.54 GB), making the
    forward+backward fit comfortably on a 14 GB GPU without autocast or
    gradient checkpointing. The upsampled gradient retains sufficient spatial
    resolution for region-level atlas projection.

    Args:
        model:              Frozen 3-D CNN (all params have requires_grad=False).
        x_tensor:           Input tensor (1, C, H, W, D), float32.
        importance_weights: (n_features,) signed weights — group-difference
                            t-statistics from Child vs Adult / PD vs HC.
                            For concatenation aggregations whose weight vector
                            is k×n_features, segments are averaged to n_features.
        device:             Compute device ('cpu' or 'cuda').
        downsample_factor:  Spatial downsampling applied before the forward pass.
                            Default 2 (256³→128³).  Use 1 to skip.

    Returns:
        signed_map: (H, W, D) float32 — mean gradient over C channels, retains sign.
        abs_map:    (H, W, D) float32 — channel-wise |gradient| averaged over channels.
    """
    str_device = str(device)
    cache = forward_cache(model, x_tensor, device=device, downsample_factor=downsample_factor)
    n_model_feat = getattr(model, "out_features", None)
    try:
        return backprop_from_w(cache, importance_weights, n_model_feat, retain_graph=False)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and "cuda" in str_device:
            torch.cuda.empty_cache()
            return compute_attribution_map(
                model, x_tensor, importance_weights, "cpu", downsample_factor
            )
        raise


def project_to_atlas(
    signed_map: np.ndarray,
    abs_map: np.ndarray,
    atlas: np.ndarray,
) -> tuple[dict[int, tuple[float, int]], dict[int, tuple[float, int]]]:
    """Average signed and absolute attribution maps within each aparc+aseg label.

    Uses np.bincount for an O(n_voxels) single pass rather than iterating over
    labels (O(n_labels × n_voxels) ≈ 150 × 16M ops). Typical speedup: ~150×.

    Args:
        signed_map: (H, W, D) float32 — signed attribution map.
        abs_map:    (H, W, D) float32 — absolute attribution map.
        atlas:      (H, W, D) int32  — FreeSurfer aparc+aseg label volume.

    Returns:
        (result_signed, result_abs) where each is
        {label_id: (mean_value, n_voxels)} for every non-zero label.
    """
    labels_flat = atlas.ravel().astype(np.intp)
    n_bins = int(labels_flat.max()) + 1

    counts = np.bincount(labels_flat, minlength=n_bins)
    sums_s = np.bincount(labels_flat, weights=signed_map.ravel().astype(np.float64), minlength=n_bins)
    sums_a = np.bincount(labels_flat, weights=abs_map.ravel().astype(np.float64), minlength=n_bins)

    nonzero_labels = np.flatnonzero(counts)[1:]  # skip label 0 (background)
    result_signed: dict[int, tuple[float, int]] = {}
    result_abs:    dict[int, tuple[float, int]] = {}
    for lbl in nonzero_labels:
        n_vox = int(counts[lbl])
        result_signed[int(lbl)] = (float(sums_s[lbl] / n_vox), n_vox)
        result_abs[int(lbl)]    = (float(sums_a[lbl] / n_vox), n_vox)
    return result_signed, result_abs


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class VoxelImportanceResult:
    """Result of a voxel importance computation.

    Attributes:
        signed_map: (H, W, D) float32 gradient map retaining sign.
            Positive voxels drive features that are *higher* in the reference
            group (positive importance_weights direction); negative voxels drive
            features that are *lower*.
        abs_map: (H, W, D) float32 unsigned sensitivity map.
            Shows total voxel contribution regardless of direction — useful for
            identifying the most spatially informative regions.
        region_df: DataFrame with one row per FreeSurfer atlas region, columns:
            label_id, region, n_voxels, signed_importance, abs_importance.
            Sorted by abs_importance descending.
    """

    signed_map: np.ndarray
    abs_map: np.ndarray
    region_df: pd.DataFrame

    def top_regions(self, n: int = 10, by: str = "abs_importance") -> pd.DataFrame:
        """Return the top-n rows of region_df sorted by `by` (default: abs_importance)."""
        return self.region_df.nlargest(n, by).reset_index(drop=True)


def compute_voxel_importance(
    model: torch.nn.Module,
    brain_volume: "np.ndarray | torch.Tensor",
    importance_weights: np.ndarray,
    atlas: np.ndarray,
    lut: "dict[int, str] | str | Path",
    *,
    device: "str | torch.device" = "cpu",
    downsample_factor: int = 2,
) -> VoxelImportanceResult:
    """End-to-end voxel importance: backpropagate weights → gradient map → region table.

    This is the recommended entry point. It wraps ``compute_attribution_map``,
    ``project_to_atlas``, and ``load_freesurfer_lut`` into a single call and
    returns a structured ``VoxelImportanceResult``.

    How it works
    ------------
    1. Form scalar  S = Σᵢ wᵢ · CNNᵢ(x)  where wᵢ are the importance_weights
       (e.g. t-statistics from a group-difference test).
    2. Compute ∂S/∂x via backpropagation through the frozen CNN.
       The input is optionally downsampled (default 2×: 256³→128³) to reduce
       peak GPU memory from ~4 GB to ~0.5 GB, then the gradient is upsampled
       back to the original resolution.
    3. Average the gradient within each brain region defined by the FreeSurfer
       aparc+aseg atlas, using the LUT to attach human-readable region names.

    Args:
        model:              Frozen 3D CNN (CNN3D_DoubleConv or CNN3D_CovPool).
                            All parameters must have requires_grad=False — call
                            ``src.models.freeze_model(model)`` if needed.
        brain_volume:       Brain scan as either:
                            - (H, W, D) numpy array (single-channel, most common),
                              automatically unsqueezed to (1, 1, H, W, D).
                            - (1, C, H, W, D) tensor (pre-batched, multi-channel).
        importance_weights: (n_features,) array of signed weights. Positive wᵢ
                            means feature i is elevated in the reference group.
                            Typically the t-statistics from ``group_diff.run_option2_ttest``.
        atlas:              (H, W, D) int32 array of aparc+aseg label IDs.
                            Must be in the same voxel space as brain_volume.
        lut:                Either a ``{label_id: region_name}`` dict (from
                            ``load_freesurfer_lut``), or a path to ``FS-LUT.txt``.
        device:             Compute device, e.g. ``"cpu"`` or ``"cuda"``.
                            Falls back to CPU automatically on OOM.
        downsample_factor:  Spatial downsampling before the forward pass.
                            Default 2 (256³→128³). Use 1 to skip.

    Returns:
        VoxelImportanceResult with .signed_map, .abs_map, and .region_df.

    Example
    -------
    >>> import nibabel as nib, numpy as np
    >>> from src.models import CNN3D_DoubleConv, freeze_model
    >>> from brainage_agg.analysis.voxel_attribution import compute_voxel_importance
    >>>
    >>> model = freeze_model(CNN3D_DoubleConv(in_channels=1))
    >>> brain  = nib.load("brain.mgz").get_fdata().astype("float32")
    >>> atlas  = nib.load("aparc+aseg.mgz").get_fdata().astype("int32")
    >>> w      = np.load("group_diff_t_stats.npy")   # shape (7680,)
    >>>
    >>> result = compute_voxel_importance(model, brain, w, atlas, "FS-LUT.txt",
    ...                                  device="cuda")
    >>> print(result.top_regions(n=10))
    >>> result.region_df.to_csv("voxel_importance.csv", index=False)
    """
    # --- Resolve LUT ---
    if not isinstance(lut, dict):
        lut = load_freesurfer_lut(lut)

    # --- Prepare input tensor ---
    if isinstance(brain_volume, np.ndarray):
        x = torch.from_numpy(brain_volume.astype(np.float32))
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # (H,W,D) → (1,1,H,W,D)
    else:
        x = brain_volume.float()
    if x.ndim != 5:
        raise ValueError(
            f"brain_volume must be (H,W,D) or (1,C,H,W,D), got shape {tuple(x.shape)}"
        )

    # --- Gradient attribution ---
    signed_map, abs_map = compute_attribution_map(
        model, x, importance_weights, device=device,
        downsample_factor=downsample_factor,
    )

    # --- Atlas projection ---
    result_signed, result_abs = project_to_atlas(signed_map, abs_map, atlas)

    # --- Build region DataFrame ---
    rows = []
    for label_id, (signed_val, n_vox) in result_signed.items():
        abs_val = result_abs[label_id][0]
        rows.append({
            "label_id":          label_id,
            "region":            lut.get(label_id, f"label_{label_id}"),
            "n_voxels":          n_vox,
            "signed_importance": signed_val,
            "abs_importance":    abs_val,
        })

    region_df = (
        pd.DataFrame(rows)
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )

    return VoxelImportanceResult(
        signed_map=signed_map,
        abs_map=abs_map,
        region_df=region_df,
    )
