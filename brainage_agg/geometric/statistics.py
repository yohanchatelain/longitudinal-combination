"""Statistical primitives for the geometric longitudinal validation decision rules.

Plan §5.3 (classical/mixed-effects analysis), §6.1-6.3 (calibration, dose-response,
magnitude-accuracy metrics), and §8 (subject-clustered bootstrap, equivalence
testing). Each function is a small, generic primitive, independently testable on
synthetic arrays; the orchestrator (not yet built) composes them into the actual
Experiment 1-6 analyses once cohort-level results exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def paired_change_test(differences: np.ndarray) -> dict:
    """Paired one-sample test of subject-level change against zero (plan §5.3).

    Reports both the parametric paired t-test and a robust Wilcoxon signed-rank
    sensitivity result side by side, since "statistical significance alone does not
    establish accuracy because the numerical truth is known" (plan §5.3) -- neither
    test alone is treated as sufficient.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        raise ValueError("paired_change_test requires at least two finite differences")
    t_stat, t_p = stats.ttest_1samp(differences, popmean=0.0)
    try:
        w_stat, w_p = stats.wilcoxon(differences)
    except ValueError:
        # All differences identical (commonly all zero): Wilcoxon is undefined.
        w_stat, w_p = float("nan"), float("nan")
    return {
        "n": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def subject_clustered_bootstrap_ci(
    values: Sequence,
    subject_ids: Sequence[str],
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for `statistic_fn`, resampled by whole subject.

    Plan §8: "Bootstrap resampling is clustered by subject." Resamples the set of
    *unique subjects* with replacement (not individual rows), then re-evaluates
    `statistic_fn` on that resampled population's rows, so within-subject structure
    (e.g. multiple conditions per subject) is preserved rather than treated as
    independent observations.
    """
    values = np.asarray(values)
    subject_ids = np.asarray([str(subject) for subject in subject_ids])
    if len(values) != len(subject_ids):
        raise ValueError("values and subject_ids must have the same length")
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("subject_clustered_bootstrap_ci requires at least two subjects")
    rng = np.random.default_rng(seed)
    point_estimate = float(statistic_fn(values))
    draws = np.empty(n_bootstrap, dtype=float)
    for draw_index in range(n_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        mask = np.concatenate([np.flatnonzero(subject_ids == subject) for subject in sampled_subjects])
        draws[draw_index] = statistic_fn(values[mask])
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point_estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "excludes_zero": bool(lower > 0 or upper < 0),
        "n_bootstrap": int(n_bootstrap),
        "n_subjects": int(len(unique_subjects)),
    }


def spearman_dose_response(dose: np.ndarray, response: np.ndarray) -> dict:
    """Spearman correlation between realized Jacobian-derived dose and response
    (plan §6.2 dose-response metrics)."""
    dose = np.asarray(dose, dtype=float)
    response = np.asarray(response, dtype=float)
    if len(dose) != len(response):
        raise ValueError("dose and response must have the same length")
    if len(dose) < 3:
        raise ValueError("spearman_dose_response requires at least three paired observations")
    correlation, p_value = stats.spearmanr(dose, response)
    return {"spearman_r": float(correlation), "p_value": float(p_value), "n": int(len(dose))}


def calibration_slope_intercept(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """OLS slope/intercept of predicted-vs-true (plan §6.2 calibration slope; §6.3
    'calibration slope and intercept'). Slope=1, intercept=0 is perfect calibration.
    """
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length")
    if len(true_values) < 2:
        raise ValueError("calibration_slope_intercept requires at least two observations")
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_values, predicted_values)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_value),
        "p_value": float(p_value),
        "slope_std_err": float(std_err),
    }


def sign_accuracy(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Fraction of observations where predicted and true change share the same sign
    (plan §6.2 'sign accuracy'). Observations with a zero true or predicted value are
    excluded, since they have no defined sign to match.
    """
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length")
    defined = (true_values != 0) & (predicted_values != 0)
    if not defined.any():
        raise ValueError("sign_accuracy requires at least one observation with a defined sign")
    matches = np.sign(true_values[defined]) == np.sign(predicted_values[defined])
    return float(np.mean(matches))


def subject_level_discrimination_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Subject-level AUROC distinguishing dosed from sham subjects by response score
    (plan §6.2 'subject-level discrimination AUROC')."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(np.unique(labels)) < 2:
        raise ValueError("subject_level_discrimination_auroc requires both classes to be present")
    return float(roc_auc_score(labels, scores))


def regression_accuracy_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """MAE, RMSE, signed bias, and calibration slope/intercept (plan §6.3 Experiment
    3 magnitude-accuracy metrics, excluding interval coverage -- see
    `interval_coverage` for that, which needs caller-supplied prediction intervals).
    """
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length")
    if len(true_values) < 1:
        raise ValueError("regression_accuracy_metrics requires at least one observation")
    errors = predicted_values - true_values
    calibration = calibration_slope_intercept(true_values, predicted_values) if len(true_values) >= 2 else None
    return {
        "n": int(len(true_values)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "signed_bias": float(np.mean(errors)),
        "calibration_slope": calibration["slope"] if calibration else float("nan"),
        "calibration_intercept": calibration["intercept"] if calibration else float("nan"),
    }


def interval_coverage(
    true_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    """Fraction of true values falling inside caller-supplied [lower, upper] interval
    estimates (plan §6.3 '95% interval coverage'). Generic over how those bounds were
    produced (e.g. a decoder's predictive interval) -- this module has no opinion on
    the interval-estimation method itself.
    """
    true_values = np.asarray(true_values, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    if not (len(true_values) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError("true_values, lower_bounds, and upper_bounds must have the same length")
    if len(true_values) < 1:
        raise ValueError("interval_coverage requires at least one observation")
    covered = (true_values >= lower_bounds) & (true_values <= upper_bounds)
    return float(np.mean(covered))


def equivalence_test_tost(differences: np.ndarray, *, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for equivalence within +/- `margin`.

    Plan §8: "If differences are small, perform an equivalence analysis using a
    prespecified margin. Failure to reject a null hypothesis is not interpreted as
    equivalence." Equivalence is concluded only if BOTH one-sided nulls (difference
    <= -margin; difference >= margin) are rejected at `alpha` -- a non-significant
    ordinary test is never substituted for this.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 2:
        raise ValueError("equivalence_test_tost requires at least two finite differences")
    if margin <= 0:
        raise ValueError("margin must be positive")
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(n))
    if standard_error == 0:
        equivalent = bool(-margin < mean < margin)
        return {
            "mean_difference": mean,
            "lower_p_value": 0.0 if equivalent else 1.0,
            "upper_p_value": 0.0 if equivalent else 1.0,
            "equivalent": equivalent,
            "n": n,
        }
    degrees_of_freedom = n - 1
    t_lower = (mean - (-margin)) / standard_error
    t_upper = (mean - margin) / standard_error
    lower_p_value = 1.0 - stats.t.cdf(t_lower, degrees_of_freedom)  # H0: mean <= -margin
    upper_p_value = stats.t.cdf(t_upper, degrees_of_freedom)  # H0: mean >= margin
    equivalent = bool(lower_p_value < alpha and upper_p_value < alpha)
    return {
        "mean_difference": mean,
        "lower_p_value": float(lower_p_value),
        "upper_p_value": float(upper_p_value),
        "equivalent": equivalent,
        "n": n,
    }


def power_curve_by_resampling(
    values: np.ndarray,
    subject_ids: Sequence[str],
    *,
    sample_sizes: Sequence[int],
    reject_null_fn: Callable[[np.ndarray], bool],
    n_resamples: int = 1000,
    seed: int,
) -> pd.DataFrame:
    """Empirical power at each sample size, by resampling completed subject-level
    results rather than rerunning the image pipeline.

    Plan §6.2: "Sample-size curves are calculated by nested resampling of completed
    results, not by rerunning the image pipeline." For each `n` in `sample_sizes`,
    draws `n` subjects with replacement from the completed pool `n_resamples` times
    and reports the fraction of draws where `reject_null_fn(drawn_values)` is True.
    """
    values = np.asarray(values, dtype=float)
    subject_ids = np.asarray([str(subject) for subject in subject_ids])
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("power_curve_by_resampling requires at least two subjects")
    if any(n < 1 for n in sample_sizes):
        raise ValueError("sample_sizes must all be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for sample_size in sample_sizes:
        rejections = 0
        for _ in range(n_resamples):
            sampled_subjects = rng.choice(unique_subjects, size=sample_size, replace=True)
            mask = np.concatenate([np.flatnonzero(subject_ids == subject) for subject in sampled_subjects])
            if reject_null_fn(values[mask]):
                rejections += 1
        rows.append({"sample_size": int(sample_size), "power": rejections / n_resamples, "n_resamples": n_resamples})
    return pd.DataFrame(rows)
