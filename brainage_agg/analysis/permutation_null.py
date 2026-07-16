"""Permutation-null architecture-bias correction for voxel attribution maps.

The frozen, untrained CNN feature extractor has its own spatial gradient
structure (architecture bias) independent of any group-difference signal in
``w``. This module estimates that bias via a permutation null —
backpropagating *permuted* weight vectors through the same cached forward
pass — so that ``S - E[S_perm]`` (and the z-map
``(S - E[S_perm]) / sqrt(Var[S_perm])``) isolate the group-difference signal
from the architecture's own bias.

Two null designs (compared in the Phase 4a pilot,
``brainage_agg.experiment.run_permutation_null_pilot``):

  index_perm  -- cheap. Cache per-feature gradients
                 ``G_j = d(feature_j)/d(x_small)`` once per (subject, seed)
                 via ``cache_topk_gradients``, then for each permutation just
                 recombine: ``S_perm = sum_j w_perm[j] * G_j`` (einsum, no
                 extra backward passes).
  group_perm  -- expensive but exact. Permute the *group labels*, re-derive
                 ``w_t_perm = compute_w_t(X_agg, perm_bands, groups)``,
                 ``rank_normalize``, and run a full ``backprop_from_w`` per
                 permutation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from brainage_agg.analysis.voxel_attribution import AttributionCache, _raw_grad_small


# ---------------------------------------------------------------------------
# Online mean/variance accumulation
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """Online mean/variance accumulator (Welford's algorithm).

    Avoids holding all ``N`` permutation maps in memory at once — each
    ``update`` folds one sample into running mean/M2 statistics.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, variance)`` as float32, variance with ddof=1.

        Variance is all-zero if fewer than 2 samples were seen.
        """
        if self.n < 2:
            return self.mean.astype(np.float32), np.zeros_like(self.M2, dtype=np.float32)
        return self.mean.astype(np.float32), (self.M2 / (self.n - 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-k feature selection and per-feature gradient caching
# ---------------------------------------------------------------------------

def topk_feature_indices(w_by_type: dict[str, np.ndarray], k: int) -> np.ndarray:
    """Union of the top-``k`` |w| feature indices across weight types.

    Returns a sorted, deduplicated `np.intp` array, length <= ``k * len(w_by_type)``.
    """
    idx: set[int] = set()
    for w in w_by_type.values():
        w = np.asarray(w)
        top = np.argsort(np.abs(w))[::-1][:k]
        idx.update(int(i) for i in top)
    return np.array(sorted(idx), dtype=np.intp)


def cache_topk_gradients(
    cache: AttributionCache,
    model_out_features: int,
    topk_idx: np.ndarray,
    retain_graph: bool = True,
) -> np.ndarray:
    """Per-feature gradients ``d(feature_j)/d(x_small)`` for ``j in topk_idx``.

    One backward pass per index (one-hot weight vector ``e_j``), at the
    cached *downsampled* resolution — no upsampling, since the
    feature-index-permutation null recombines these per permutation and only
    upsamples the final Welford mean/var maps once (see
    ``run_permutation_null_pilot``).

    Args:
        cache:               Result of ``forward_cache``.
        model_out_features:  Length of ``cache.features[0]``.
        topk_idx:            Feature indices to cache gradients for, e.g.
                             from ``topk_feature_indices``. Assumed to index
                             ``cache.features[0]`` directly — true for
                             non-concatenation aggregations (e.g. "mean").
                             Concatenation aggregations would need
                             ``topk_idx % model_out_features`` first (not
                             exercised by the Phase 4a pilot).
        retain_graph:        Forwarded to the *last* of the ``len(topk_idx)``
                             backward passes (all earlier ones always retain,
                             since they are not the cache's last backward
                             pass). Pass ``False`` only if no further
                             backward passes will run on this ``cache``.

    Returns:
        G, shape ``(k, C, h, w, d)`` float32.
    """
    topk_idx = np.asarray(topk_idx, dtype=np.intp)
    n = len(topk_idx)
    grads = np.empty((n, *cache.x_small.shape[1:]), dtype=np.float32)
    for i, j in enumerate(topk_idx):
        w = np.zeros(model_out_features, dtype=np.float32)
        w[int(j)] = 1.0
        is_last = i == n - 1
        grad_small = _raw_grad_small(
            cache, w, model_out_features,
            retain_graph=True if not is_last else retain_graph,
        )
        grads[i] = grad_small.cpu().numpy()
    return grads


# ---------------------------------------------------------------------------
# Null correction
# ---------------------------------------------------------------------------

def null_correction(
    observed_map: np.ndarray,
    null_mean: np.ndarray,
    null_var: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Z-map: ``(observed_map - null_mean) / sqrt(null_var + eps)``."""
    return (observed_map - null_mean) / np.sqrt(null_var + eps)


def derive_threshold(z_map: np.ndarray, alpha: float = 0.05) -> float:
    """Two-sided z threshold for ``alpha``, Bonferroni-corrected over ``z_map``.

    Conservative voxelwise FWER control: ``alpha / (2 * n_voxels)`` per tail,
    where ``n_voxels = z_map.size``. Cluster-level alternatives can be layered
    on top in Phase 4b if this proves too conservative.
    """
    n_voxels = z_map.size
    return float(norm.ppf(1.0 - alpha / (2.0 * n_voxels)))
