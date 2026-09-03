"""Regularized linear decoder: feature-change -> %volume-change (plan §6.3).

"A secondary analysis fits a regularized linear decoder from frozen feature changes
to percentage volume change. The decoder is trained only on separate synthetic
calibration subjects and evaluated on untouched validation subjects. This method is
described as a frozen random encoder with a calibrated linear readout, not as a
wholly untrained pipeline" (plan §6.3). This module never accepts a "validation"
argument by design, so validation data cannot leak into fitting even by mistake --
the caller passes calibration data here and evaluates predictions against validation
data separately, with `statistics.regression_accuracy_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold


@dataclass(frozen=True)
class DecoderFitResult:
    alpha: float
    model: Ridge
    n_calibration_subjects: int


def select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: list[float],
    n_folds: int = 5,
) -> float:
    """Group-aware inner-CV alpha selection -- the same pattern as
    `modeling/cv.py::_select_alpha` -- grouped by subject so a subject with multiple
    calibration conditions is never split across CV folds.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("select_ridge_alpha requires at least two calibration subjects")
    n_splits = min(n_folds, len(unique_groups))
    search = GridSearchCV(
        Ridge(), {"alpha": alphas}, cv=GroupKFold(n_splits=n_splits),
        scoring="neg_mean_absolute_error", refit=False,
    )
    search.fit(X, y, groups=groups)
    return float(search.best_params_["alpha"])


def fit_decoder(
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    subject_ids_calibration,
    *,
    alphas: list[float],
    n_folds: int = 5,
) -> DecoderFitResult:
    """Fit the calibrated linear readout on calibration subjects only."""
    X_calibration = np.asarray(X_calibration, dtype=float)
    y_calibration = np.asarray(y_calibration, dtype=float)
    groups = np.asarray([str(subject) for subject in subject_ids_calibration])
    if not (len(X_calibration) == len(y_calibration) == len(groups)):
        raise ValueError(
            "X_calibration, y_calibration, and subject_ids_calibration must have the same length"
        )
    alpha = select_ridge_alpha(X_calibration, y_calibration, groups, alphas=alphas, n_folds=n_folds)
    model = Ridge(alpha=alpha)
    model.fit(X_calibration, y_calibration)
    return DecoderFitResult(alpha=alpha, model=model, n_calibration_subjects=int(len(np.unique(groups))))


def predict(result: DecoderFitResult, X: np.ndarray) -> np.ndarray:
    return np.asarray(result.model.predict(np.asarray(X, dtype=float)), dtype=float)


def bootstrap_prediction_intervals(
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    subject_ids_calibration,
    X_query: np.ndarray,
    *,
    alpha: float,
    n_bootstrap: int = 500,
    seed: int,
    interval: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-bootstrap prediction interval for `X_query`.

    Resampled by whole calibration subject -- the same clustering discipline as
    `statistics.subject_clustered_bootstrap_ci` -- refitting a fixed-alpha Ridge on
    each resample. Feeds plan §6.3's "95% interval coverage" metric via
    `statistics.interval_coverage`.
    """
    X_calibration = np.asarray(X_calibration, dtype=float)
    y_calibration = np.asarray(y_calibration, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    groups = np.asarray([str(subject) for subject in subject_ids_calibration])
    unique_subjects = np.unique(groups)
    if len(unique_subjects) < 2:
        raise ValueError("bootstrap_prediction_intervals requires at least two calibration subjects")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, len(X_query)), dtype=float)
    for draw_index in range(n_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == subject) for subject in sampled_subjects])
        model = Ridge(alpha=alpha)
        model.fit(X_calibration[mask], y_calibration[mask])
        draws[draw_index] = model.predict(X_query)
    lower_pct = 100 * (1 - interval) / 2
    upper_pct = 100 * (interval + (1 - interval) / 2)
    lower_bounds = np.percentile(draws, lower_pct, axis=0)
    upper_bounds = np.percentile(draws, upper_pct, axis=0)
    return lower_bounds, upper_bounds
