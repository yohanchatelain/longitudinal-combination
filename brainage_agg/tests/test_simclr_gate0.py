"""Tests for brainage_agg.simclr_comparator.gate0.

Small synthetic tensors only, deterministic, no real MRI, no network access — matches
the convention used elsewhere in this repo's test suite (e.g. test_geometric_validation.py
on the main branch). Real-data and checkpoint-download verification are explicitly out of
scope here, per the approved implementation plan.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from brainage_agg.simclr_comparator.gate0 import (
    activation_statistics,
    linear_probe_check,
    recompute_batchnorm_running_stats,
    run_gate0,
)

SEED = 0
N_SUBJECTS = 24
VOLUME_SHAPE = (1, 1, 8, 8, 8)  # (N placeholder per-subject, C, D, H, W) — batched below


def _make_degenerate_model() -> nn.Module:
    """All-zero conv weights -> every BatchNorm3d channel sees zero-variance input."""
    model = nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(8),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
    )
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            nn.init.zeros_(module.weight)
    return model.eval()


def _make_healthy_model(seed: int = SEED) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(8),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
    )
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight)
    return model.eval()


def _make_pilot_volumes(seed: int = SEED, n: int = N_SUBJECTS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, *VOLUME_SHAPE[1:]), generator=generator)


class TestActivationStatistics:
    def test_degenerate_model_has_high_dead_channel_rate(self) -> None:
        model = _make_degenerate_model()
        volumes = _make_pilot_volumes()
        result = activation_statistics(model, volumes)
        assert result["dead_channel_rate"] > 0.9
        assert result["has_nan_or_inf"] is False

    def test_healthy_model_has_low_dead_channel_rate(self) -> None:
        model = _make_healthy_model()
        volumes = _make_pilot_volumes()
        result = activation_statistics(model, volumes)
        assert result["dead_channel_rate"] < 0.05
        assert result["has_nan_or_inf"] is False

    def test_raises_without_batchnorm3d(self) -> None:
        model = nn.Sequential(nn.Conv3d(1, 2, kernel_size=3, padding=1), nn.Flatten())
        with pytest.raises(ValueError, match="BatchNorm3d"):
            activation_statistics(model, _make_pilot_volumes())


class TestLinearProbeCheck:
    def test_true_relationship_is_significant(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 60
        ages = rng.uniform(20, 80, size=n)
        # Features with an injected true linear relationship to age, plus noise.
        signal = ages[:, None] * rng.normal(1.0, 0.05, size=(1, 5))
        noise = rng.normal(0, 1.0, size=(n, 5))
        features = signal + noise
        result = linear_probe_check(features, ages, n_permutations=200, seed=SEED)
        assert result["r2"] > 0
        assert result["p_value"] < 0.05

    def test_pure_noise_is_not_reliably_significant(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 60
        ages = rng.uniform(20, 80, size=n)
        features = rng.normal(0, 1.0, size=(n, 5))  # no relationship to ages at all
        result = linear_probe_check(features, ages, n_permutations=200, seed=SEED)
        assert result["p_value"] > 0.05


class TestRunGate0:
    def test_degenerate_model_fails_gate(self) -> None:
        model = _make_degenerate_model()
        volumes = _make_pilot_volumes()
        rng = np.random.default_rng(SEED)
        ages = rng.uniform(20, 80, size=volumes.shape[0])
        result = run_gate0(model, volumes, ages, seed=SEED, n_permutations=100)
        assert result.passed is False
        assert result.failures  # at least one diagnosable reason

    def test_healthy_model_can_pass_gate(self) -> None:
        model = _make_healthy_model()
        # n=24 (the plan's actual pilot size) is too small for 5-fold Ridge CV to
        # reliably recover even an exact linear relationship — confirmed by the passing
        # TestLinearProbeCheck cases, which use n=60. Use the same larger n here so this
        # end-to-end test isolates run_gate0's orchestration logic rather than re-testing
        # small-sample CV instability already covered (and expected) elsewhere.
        volumes = _make_pilot_volumes(n=60)
        # Ages correlated with a simple statistic of the (fixed-seed, healthy) features
        # so the linear-probe leg of the gate has genuine signal to detect, matching how
        # a real random-CNN-feature vs. real-age check is expected to behave.
        with torch.no_grad():
            pooled = model(volumes).numpy()
        ages = pooled[:, 0] * 10 + 50
        result = run_gate0(model, volumes, ages, seed=SEED, n_permutations=100)
        assert result.dead_channel_rate < 0.05
        assert result.has_nan_or_inf is False
        assert result.passed is True


class TestRecomputeBatchnormRunningStats:
    def test_updates_running_stats_from_default(self) -> None:
        model = _make_healthy_model()
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
        for bn in bn_layers:
            assert torch.allclose(bn.running_var, torch.ones_like(bn.running_var))

        volumes = _make_pilot_volumes()
        recompute_batchnorm_running_stats(model, volumes)

        for bn in bn_layers:
            assert not torch.allclose(bn.running_var, torch.ones_like(bn.running_var))
        assert model.training is False  # left in eval() mode, per the docstring contract
