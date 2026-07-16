"""Tests for `permutation_null` and the `_raw_grad_small` refactor.

All tests use tiny synthetic tensors/models (no MRI data, no RF/SHAP) — safe
to run on the login node in well under a second.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import backprop_from_w, forward_cache
from brainage_agg.analysis.permutation_null import (
    WelfordAccumulator,
    cache_topk_gradients,
    derive_threshold,
    null_correction,
    topk_feature_indices,
)


# ---------------------------------------------------------------------------
# WelfordAccumulator
# ---------------------------------------------------------------------------

def test_welford_matches_numpy():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((30, 4, 5))

    acc = WelfordAccumulator(shape=(4, 5))
    for s in samples:
        acc.update(s)
    mean, var = acc.finalize()

    np.testing.assert_allclose(mean, samples.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(var, samples.var(axis=0, ddof=1), atol=1e-6)


def test_welford_single_sample_has_zero_variance():
    acc = WelfordAccumulator(shape=(3,))
    acc.update(np.array([1.0, 2.0, 3.0]))
    mean, var = acc.finalize()

    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(var, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# topk_feature_indices
# ---------------------------------------------------------------------------

def test_topk_feature_indices_union_and_sorted():
    w_t = np.array([0.1, -5.0, 0.2, 3.0, 0.0])
    w_shap = np.array([4.0, 0.1, -0.2, 0.05, 1.0])

    idx = topk_feature_indices({"tstat": w_t, "shap": w_shap}, k=2)

    # top-2 |w_t| -> {1, 3}; top-2 |w_shap| -> {0, 4}
    assert idx.tolist() == [0, 1, 3, 4]
    assert idx.dtype == np.intp


# ---------------------------------------------------------------------------
# cache_topk_gradients vs. a linear model's known gradient
# ---------------------------------------------------------------------------

class _TinyLinear(torch.nn.Module):
    """y = W @ flatten(x), no bias -- gradient of y_j w.r.t. x is just W[j]."""

    def __init__(self, in_shape: tuple[int, int, int, int], out_features: int):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        n_in = int(np.prod(in_shape))
        self.linear = torch.nn.Linear(n_in, out_features, bias=False)
        self.out_features = out_features

    def forward(self, x):
        return self.linear(self.flatten(x))


def test_cache_topk_gradients_matches_linear_weights():
    torch.manual_seed(0)
    c, h, w, d = 1, 2, 2, 2
    out_features = 6
    model = _TinyLinear((c, h, w, d), out_features).eval()
    x = torch.randn(1, c, h, w, d)

    cache = forward_cache(model, x, device="cpu", downsample_factor=1)

    topk_idx = np.array([0, 2, 5])
    G = cache_topk_gradients(cache, out_features, topk_idx, retain_graph=False)

    assert G.shape == (3, c, h, w, d)
    weight = model.linear.weight.detach().numpy()  # (out_features, n_in)
    for i, j in enumerate(topk_idx):
        expected = weight[j].reshape(c, h, w, d)
        np.testing.assert_allclose(G[i], expected, atol=1e-6)


def test_backprop_from_w_unchanged_after_refactor():
    """Smoke test: backprop_from_w still returns finite (H,W,D) maps."""
    torch.manual_seed(1)
    c, h, w, d = 1, 4, 4, 4
    out_features = 5
    model = _TinyLinear((c, h, w, d), out_features).eval()
    x = torch.randn(1, c, h, w, d)

    cache = forward_cache(model, x, device="cpu", downsample_factor=1)
    weights = np.array([1.0, -1.0, 0.5, 0.0, 2.0], dtype=np.float32)
    signed_map, abs_map = backprop_from_w(cache, weights, out_features, retain_graph=False)

    assert signed_map.shape == (h, w, d)
    assert abs_map.shape == (h, w, d)
    assert np.isfinite(signed_map).all()
    assert np.isfinite(abs_map).all()
    assert (abs_map >= 0).all()


# ---------------------------------------------------------------------------
# null_correction / derive_threshold
# ---------------------------------------------------------------------------

def test_null_correction_zscores():
    observed = np.array([2.0, 5.0])
    null_mean = np.array([0.0, 1.0])
    null_var = np.array([1.0, 4.0])

    z = null_correction(observed, null_mean, null_var)
    np.testing.assert_allclose(z, [2.0, 2.0], atol=1e-6)


def test_derive_threshold_bonferroni():
    from scipy.stats import norm

    z_map = np.zeros((10, 10, 10))  # 1000 voxels
    threshold = derive_threshold(z_map, alpha=0.05)

    expected = norm.ppf(1.0 - 0.05 / (2 * 1000))
    assert np.isclose(threshold, expected)
    assert threshold > norm.ppf(0.975)  # stricter than uncorrected p=0.05
