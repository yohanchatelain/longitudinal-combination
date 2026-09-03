"""Permutation-preserving regional inference and maxT calibration."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def empirical_region_pvalues(
    observed: np.ndarray,
    permutations: np.ndarray,
    *,
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray]:
    """Return pointwise and max-statistic FWER empirical p-values.

    ``permutations`` is never collapsed: its required shape is
    ``(n_permutations, n_regions)``. The standard +1 correction guarantees
    nonzero valid p-values under Monte Carlo sampling.
    """
    observed = np.asarray(observed, dtype=float)
    permutations = np.asarray(permutations, dtype=float)
    if observed.ndim != 1 or permutations.ndim != 2 or permutations.shape[1] != len(observed):
        raise ValueError("Expected observed=(regions,) and permutations=(permutations, regions)")
    if permutations.shape[0] < 2:
        raise ValueError("At least two permutations are required")
    if not np.isfinite(observed).all() or not np.isfinite(permutations).all():
        raise ValueError("Observed and permutation statistics must all be finite")
    if alternative == "greater":
        observed_test = observed
        permutation_test = permutations
    elif alternative == "less":
        observed_test = -observed
        permutation_test = -permutations
    elif alternative == "two-sided":
        observed_test = np.abs(observed)
        permutation_test = np.abs(permutations)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    denominator = permutations.shape[0] + 1
    pointwise = (1 + np.sum(permutation_test >= observed_test[None, :], axis=0)) / denominator
    permutation_max = np.max(permutation_test, axis=1)
    adjusted = (1 + np.sum(permutation_max[:, None] >= observed_test[None, :], axis=0)) / denominator
    return pointwise.astype(float), adjusted.astype(float)


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson interval, including boundary cases."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def maxT_pseudo_null_calibration(
    permutations: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Leave-one-permutation-out estimate of maxT family-wise false positives."""
    permutations = np.asarray(permutations, dtype=float)
    if permutations.ndim != 2 or permutations.shape[0] < 3:
        raise ValueError("Need at least three permutation-indexed regional-statistic rows")
    false_positive = np.zeros(permutations.shape[0], dtype=bool)
    for index in range(permutations.shape[0]):
        null = np.delete(permutations, index, axis=0)
        _, adjusted = empirical_region_pvalues(permutations[index], null, alternative="greater")
        false_positive[index] = bool(np.any(adjusted <= alpha))
    successes = int(false_positive.sum())
    lower, upper = binomial_interval(successes, len(false_positive))
    return {
        "n_pseudo_null": int(len(false_positive)),
        "n_familywise_false_positive": successes,
        "maxT_fwer": float(false_positive.mean()),
        "maxT_fwer_ci_lower": lower,
        "maxT_fwer_ci_upper": upper,
    }
