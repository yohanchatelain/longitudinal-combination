from __future__ import annotations

import numpy as np

from registration_free_fixed_features.config import EvaluationConfig
from registration_free_fixed_features.evaluation import make_outer_folds, run_paired_nested_cv
from registration_free_fixed_features.longitudinal import VisitFeature, build_subject_representations


def test_two_visit_representations_are_non_redundant_and_named():
    first = np.array([1.0, 3.0], dtype=np.float32)
    last = np.array([3.0, 7.0], dtype=np.float32)
    representations = build_subject_representations(
        [VisitFeature(0.0, first), VisitFeature(2.0, last)],
        ("a", "b"),
        min_elapsed_years=0.25,
        max_elapsed_years=10.0,
    )
    assert set(representations) == {"baseline", "mean_rate", "rate"}
    np.testing.assert_allclose(representations["mean_rate"].values, [2, 5, 1, 2])
    assert representations["mean_rate"].names == (
        "endpoint_mean|a",
        "endpoint_mean|b",
        "annualized_rate|a",
        "annualized_rate|b",
    )


def test_multi_visit_ols_recovers_intercept_and_slope():
    visits = [
        VisitFeature(time, np.array([2 + 3 * time, -1 + 0.5 * time]))
        for time in (0.0, 1.0, 2.5)
    ]
    output = build_subject_representations(
        visits,
        ("a", "b"),
        min_elapsed_years=0.25,
        max_elapsed_years=10.0,
    )["intercept_slope"]
    np.testing.assert_allclose(output.values, [2, -1, 3, 0.5], atol=1e-6)


def test_paired_nested_cv_predicts_every_subject_once_without_overlap():
    rng = np.random.default_rng(7)
    n_subjects = 36
    subjects = np.asarray([f"sub-{index:03d}" for index in range(n_subjects)])
    signal = rng.normal(size=(n_subjects, 3))
    y = 4 * signal[:, 0] - 2 * signal[:, 1] + rng.normal(scale=0.1, size=n_subjects)
    candidates = {
        "baseline": rng.normal(size=(n_subjects, 2)),
        "mean_rate": signal,
    }
    config = EvaluationConfig(
        outer_folds=3,
        inner_folds=2,
        random_seed=11,
        elastic_net_alphas=(0.001, 0.01),
        elastic_net_l1_ratios=(0.1, 0.9),
    )
    folds = make_outer_folds(subjects, y, task="regression", n_splits=3, random_seed=11)
    result = run_paired_nested_cv(
        candidates,
        y,
        subjects,
        task="regression",
        config=config,
        fold_ids=folds,
    )
    assert len(result.predictions) == n_subjects
    assert not result.predictions["subject_id"].duplicated().any()
    assert set(result.predictions["subject_id"]) == set(subjects)
    assert (result.fold_metrics["selected_representation"] == "mean_rate").all()
    assert result.fold_metrics["mae"].mean() < 0.5
