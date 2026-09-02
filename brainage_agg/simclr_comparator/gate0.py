"""Gate 0: random-init validity check for SimCLR(pretrained=False).

Implements simclr_experimental_plan.md §2.1-§2.2 exactly. `SimCLR`'s backbone uses
`nn.BatchNorm3d` throughout with default `track_running_stats=True` (confirmed by
inspecting the cloned model repo). A naive `pretrained=False` reinit leaves BatchNorm
running statistics at PyTorch's defaults (mean 0, variance 1), never adapted to the
activation distributions random convolution weights actually produce — a known failure
mode for random CNNs. This module decides, with numbers rather than assumption, whether
that failure mode actually bites here before any trained-vs-untrained comparison (§5)
is allowed to proceed.

This gate must pass before Experiment 5 (the mechanism experiment) runs. It is not
optional and not deferrable, per the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

DEAD_CHANNEL_VARIANCE_THRESHOLD = 1e-6
DEAD_CHANNEL_RATE_MAX = 0.05
LINEAR_PROBE_ALPHA = 0.05


@dataclass
class Gate0Result:
    """Everything needed to both decide pass/fail and diagnose a failure."""

    passed: bool
    dead_channel_rate: float
    has_nan_or_inf: bool
    probe_r2: float
    probe_p_value: float
    failures: list[str] = field(default_factory=list)


def activation_statistics(model: nn.Module, volumes: torch.Tensor) -> dict[str, float | bool]:
    """Per-channel BatchNorm3d activation variance across a forward pass over `volumes`.

    `volumes` is a single batched tensor (N, C, D, H, W) — the pilot set forward-passed
    together, not looped subject-by-subject, so hook-captured activations reflect one
    consistent forward call.

    Returns `dead_channel_rate` (fraction of BatchNorm3d channels, pooled across every
    such layer in the model, with activation variance below
    `DEAD_CHANNEL_VARIANCE_THRESHOLD`) and `has_nan_or_inf` (True if the model's final
    output contains any non-finite value).
    """
    channel_variances: list[np.ndarray] = []
    hooks = []

    def _record(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        # (N, C, D, H, W) -> variance per channel, pooled over batch and spatial dims.
        variance = output.detach().var(dim=(0, 2, 3, 4), unbiased=False)
        channel_variances.append(variance.cpu().numpy())

    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            hooks.append(module.register_forward_hook(_record))

    try:
        with torch.no_grad():
            output = model(volumes)
    finally:
        for hook in hooks:
            hook.remove()

    if not channel_variances:
        raise ValueError("No nn.BatchNorm3d layers found — activation_statistics expects at least one.")

    all_variances = np.concatenate(channel_variances)
    dead_channel_rate = float(np.mean(all_variances < DEAD_CHANNEL_VARIANCE_THRESHOLD))
    has_nan_or_inf = bool(not torch.isfinite(output).all())

    return {"dead_channel_rate": dead_channel_rate, "has_nan_or_inf": has_nan_or_inf}


def _permutation_p_value(
    observed: float, null_values: np.ndarray, *, alternative: str = "greater"
) -> float:
    """Empirical p-value of `observed` against a permutation null distribution.

    Mirrors the observed-vs-permutations pattern used in
    brainage_agg/validation/statistics.py::empirical_region_pvalues (not imported
    directly — that package is not present on this worktree's branch, which predates
    it). `alternative="greater"` matches the R² use case: a real relationship should
    score higher than a shuffled-label null, so the p-value is the fraction of null
    draws at least as extreme as observed, with a +1/+1 correction so p is never
    reported as exactly zero from a finite permutation sample.
    """
    if alternative != "greater":
        raise NotImplementedError("Only the 'greater' alternative is needed here.")
    n = null_values.shape[0]
    return float((np.sum(null_values >= observed) + 1) / (n + 1))


def linear_probe_check(
    features: np.ndarray,
    ages: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int,
    n_splits: int = 5,
) -> dict[str, float]:
    """Ridge-regression age probe + permutation test, per plan §2.1 item 2.

    Not expected to approach a trained model's accuracy — it only needs to confirm the
    untrained features carry *some* anatomical signal (consistent with Saxe et al.,
    cited in the un-CNN paper, on untrained CNNs extracting non-trivial structure).
    5-fold CV mirrors un-CNN's own evaluation protocol per the plan.
    """
    rng = np.random.default_rng(seed)

    def _cv_r2(y: np.ndarray) -> float:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        predictions = np.empty_like(y, dtype=float)
        for train_idx, test_idx in kfold.split(features):
            model = Ridge()
            model.fit(features[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
        residual = np.sum((y - predictions) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - residual / total) if total > 0 else float("nan")

    observed_r2 = _cv_r2(ages)

    null_r2 = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(ages)
        null_r2[i] = _cv_r2(shuffled)

    p_value = _permutation_p_value(observed_r2, null_r2, alternative="greater")
    return {"r2": observed_r2, "p_value": p_value}


def run_gate0(
    model: nn.Module,
    volumes: torch.Tensor,
    ages: np.ndarray,
    *,
    seed: int,
    n_permutations: int = 1000,
) -> Gate0Result:
    """Orchestrate the full §2.2 decision: activation stats + linear probe -> pass/fail.

    `model` must already be in eval mode with the weights under test (pretrained=False,
    or a post-`recompute_batchnorm_running_stats` retry) — this function does not build
    or modify the model itself.
    """
    with torch.no_grad():
        pooled_features = model(volumes).cpu().numpy()

    activation = activation_statistics(model, volumes)
    probe = linear_probe_check(pooled_features, ages, n_permutations=n_permutations, seed=seed)

    failures = []
    if activation["dead_channel_rate"] >= DEAD_CHANNEL_RATE_MAX:
        failures.append(
            f"dead_channel_rate {activation['dead_channel_rate']:.3f} >= {DEAD_CHANNEL_RATE_MAX}"
        )
    if activation["has_nan_or_inf"]:
        failures.append("NaN/Inf present in pooled output features")
    if not (probe["r2"] > 0 and probe["p_value"] < LINEAR_PROBE_ALPHA):
        failures.append(
            f"linear-probe check failed: r2={probe['r2']:.4f}, p={probe['p_value']:.4f} "
            f"(need r2 > 0 and p < {LINEAR_PROBE_ALPHA})"
        )

    return Gate0Result(
        passed=not failures,
        dead_channel_rate=activation["dead_channel_rate"],
        has_nan_or_inf=activation["has_nan_or_inf"],
        probe_r2=probe["r2"],
        probe_p_value=probe["p_value"],
        failures=failures,
    )


def recompute_batchnorm_running_stats(model: nn.Module, volumes: torch.Tensor) -> nn.Module:
    """The documented §2.2 fallback fix.

    Runs forward passes in train() mode over the pilot set so BatchNorm3d accumulates
    real running statistics from actual random-conv activations, instead of sitting at
    PyTorch's untouched defaults (mean 0, var 1). Returns the same model, mutated
    in-place and switched back to eval() — callers should re-run run_gate0() on the
    returned model. If it still fails after this, per §2.2: SimCLR-reinit is not usable
    as the untrained arm; that is a result to report, not a bug to keep chasing.
    """
    model.train()
    with torch.no_grad():
        model(volumes)
    model.eval()
    return model
