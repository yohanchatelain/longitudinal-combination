from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EvaluationConfig


Task = Literal["regression", "classification"]


@dataclass(frozen=True)
class NestedEvaluation:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame


def _regression_strata(y: np.ndarray, n_splits: int) -> np.ndarray | None:
    n_bins = min(5, max(2, len(y) // n_splits))
    try:
        bins = pd.qcut(pd.Series(y).rank(method="first"), q=n_bins, labels=False).to_numpy()
    except ValueError:
        return None
    counts = np.bincount(bins)
    return bins if counts.size and counts.min() >= n_splits else None


def make_outer_folds(
    subject_ids: np.ndarray,
    y: np.ndarray,
    *,
    task: Task,
    n_splits: int,
    random_seed: int,
) -> np.ndarray:
    subjects = np.asarray(subject_ids).astype(str)
    y = np.asarray(y)
    if len(subjects) != len(y) or len(np.unique(subjects)) != len(subjects):
        raise ValueError("Outer-fold input must contain exactly one row per unique subject")
    if n_splits > len(subjects):
        raise ValueError("More outer folds requested than subjects")
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        stratified = counts.size >= 2 and counts.min() >= n_splits
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            if stratified
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        )
        split_y = y if stratified else None
    else:
        strata = _regression_strata(y.astype(float), n_splits)
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            if strata is not None
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        )
        split_y = strata
    assignments = np.full(len(subjects), -1, dtype=int)
    dummy = np.zeros((len(subjects), 1), dtype=float)
    for fold, (_, test_indices) in enumerate(splitter.split(dummy, split_y)):
        assignments[test_indices] = fold
    if np.any(assignments < 0):
        raise RuntimeError("Outer-fold assignment is incomplete")
    return assignments


def _inner_splitter(y: np.ndarray, task: Task, requested: int, seed: int):
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.size < 2 or counts.min() < 2:
            raise ValueError("Inner classification training set lacks two samples per class")
        n_splits = min(requested, int(counts.min()))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_splits = min(requested, len(y))
    if n_splits < 2:
        raise ValueError("Inner regression training set is too small")
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _search(
    X: np.ndarray,
    y: np.ndarray,
    task: Task,
    config: EvaluationConfig,
    seed: int,
) -> GridSearchCV:
    base_steps = [("variance", VarianceThreshold()), ("scale", StandardScaler())]
    if task == "regression":
        model = ElasticNet(max_iter=20_000, random_state=seed)
        parameter_grid = {
            "model__alpha": list(config.elastic_net_alphas),
            "model__l1_ratio": list(config.elastic_net_l1_ratios),
        }
        scoring = "neg_mean_absolute_error"
    else:
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10_000,
            random_state=seed,
        )
        parameter_grid = {
            "model__C": list(config.logistic_cs),
            "model__l1_ratio": list(config.elastic_net_l1_ratios),
        }
        scoring = "balanced_accuracy"
    pipeline = Pipeline([*base_steps, ("model", model)])
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        cv=_inner_splitter(y, task, config.inner_folds, seed),
        scoring=scoring,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search


def run_paired_nested_cv(
    candidates: dict[str, np.ndarray],
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    task: Task,
    config: EvaluationConfig,
    fold_ids: np.ndarray | None = None,
) -> NestedEvaluation:
    if not candidates:
        raise ValueError("At least one feature candidate is required")
    subjects = np.asarray(subject_ids).astype(str)
    y = np.asarray(y)
    if any(matrix.ndim != 2 or len(matrix) != len(subjects) for matrix in candidates.values()):
        raise ValueError("Every candidate matrix must have one row per subject")
    folds = (
        make_outer_folds(
            subjects,
            y,
            task=task,
            n_splits=config.outer_folds,
            random_seed=config.random_seed,
        )
        if fold_ids is None
        else np.asarray(fold_ids, dtype=int)
    )
    if len(folds) != len(subjects) or np.any(folds < 0):
        raise ValueError("Invalid saved outer-fold assignments")

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if not set(subjects[train]).isdisjoint(set(subjects[test])):
            raise AssertionError(f"Subject leakage in outer fold {fold}")
        searches: dict[str, GridSearchCV] = {}
        for name, matrix in sorted(candidates.items()):
            searches[name] = _search(
                np.asarray(matrix[train], dtype=np.float64),
                y[train],
                task,
                config,
                config.random_seed + int(fold),
            )
        selected_name = max(searches, key=lambda name: searches[name].best_score_)
        selected = searches[selected_name]
        X_test = np.asarray(candidates[selected_name][test], dtype=np.float64)
        predicted = selected.predict(X_test)
        if task == "regression":
            metrics = {
                "mae": float(mean_absolute_error(y[test], predicted)),
                "r2": float(r2_score(y[test], predicted)) if test.sum() > 1 else float("nan"),
            }
            scores = np.full(test.sum(), np.nan)
        else:
            metrics = {
                "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
            }
            scores = selected.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = (
                float(roc_auc_score(y[test], scores))
                if len(np.unique(y[test])) == 2
                else float("nan")
            )
        metric_rows.append(
            {
                "fold": int(fold),
                "selected_representation": selected_name,
                "inner_score": float(selected.best_score_),
                "best_params": str(selected.best_params_),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                **metrics,
            }
        )
        for subject, truth, prediction, score in zip(
            subjects[test], y[test], predicted, scores
        ):
            prediction_rows.append(
                {
                    "subject_id": subject,
                    "fold": int(fold),
                    "y_true": float(truth),
                    "y_pred": float(prediction),
                    "score": float(score),
                    "selected_representation": selected_name,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    if set(predictions["subject_id"]) != set(subjects) or predictions["subject_id"].duplicated().any():
        raise RuntimeError("Each subject must receive exactly one outer-fold prediction")
    return NestedEvaluation(
        fold_metrics=pd.DataFrame(metric_rows),
        predictions=predictions,
        fold_assignments=pd.DataFrame({"subject_id": subjects, "fold": folds}),
    )
