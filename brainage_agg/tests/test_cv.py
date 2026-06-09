"""Cross-validation correctness tests.

Covers:
- Negative control: pure-noise Δfeature → Arm B difference model performs at chance
- Positive control: Δfeature ∝ brain_change_rate → Arm B difference model recovers
  brain_change_rate
- Chance baseline: state inputs (mean) for brain_change_rate target → R² ≈ 0
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))

from brainage_agg.modeling.cv import run_nested_cv


# ---------------------------------------------------------------------------
# Synthetic cohort factory
# ---------------------------------------------------------------------------

def _make_cohort(
    n_subjects: int,
    n_features: int,
    delta_feature_fn,
    seed: int = 0,
    brain_change_rate_fn=None,
) -> tuple[pd.DataFrame, dict]:
    """Build a synthetic cohort where Δfeature and brain_change_rate are controlled.

    delta_feature_fn(brain_change_rate, delta_t, rng) -> np.ndarray (n_features,)

    brain_change_rate_fn(rng) -> float
      Generates the per-subject Arm B target.  Defaults to N(0,1) (no signal).
    """
    rng = np.random.default_rng(seed)
    rows = []
    subject_data = {}

    for i in range(n_subjects):
        sid = f"S{i:04d}"
        delta_t = float(rng.uniform(0.5, 3.0))
        age_first = float(rng.uniform(20, 65))
        ages = np.array([age_first, age_first + delta_t])
        band = "Child" if rng.random() < 0.5 else "Adult"

        bcr = brain_change_rate_fn(rng) if brain_change_rate_fn is not None else float(rng.standard_normal())

        f_first = rng.standard_normal(n_features).astype(np.float32)
        f_last = f_first + delta_feature_fn(bcr, delta_t, rng).astype(np.float32)
        features = np.stack([f_first, f_last], axis=0)  # (2, n_features)

        rows.append(
            {
                "subject_id": sid,
                "n_timepoints": 2,
                "band": band,
                "delta_t": delta_t,
                "brain_change_rate": bcr,
                "ages": list(ages),
                "row_indices": [i * 2, i * 2 + 1],
                "t_first_idx": i * 2,
                "t_last_idx": i * 2 + 1,
                "t_mid_idx": -1,  # no mid for 2-tp subjects
                "cohort_all": True,
                "cohort_concat": False,
                "exclude_arm_b": False,
            }
        )
        subject_data[sid] = {
            "features": features,
            "ages": ages,
            "row_indices": [i * 2, i * 2 + 1],
        }

    return pd.DataFrame(rows), subject_data


# ---------------------------------------------------------------------------
# Negative control: pure-noise Δfeature → chance performance
# ---------------------------------------------------------------------------

def test_arm_b_noise_features_chance_mae():
    """When Δfeature is pure noise (uncorrelated with brain_change_rate), the Arm B
    difference model should perform at chance: R² ≈ 0.
    """
    n_subjects = 80
    n_features = 30

    def noise_delta(bcr, delta_t, rng):
        return rng.standard_normal(n_features)

    manifest, subject_data = _make_cohort(n_subjects, n_features, noise_delta, seed=7)

    results = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation="difference",
        arm="B",
        band="all",
        n_outer_folds=5,
        n_inner_folds=3,
        ridge_alphas=[0.1, 1.0, 10.0, 100.0],
    )
    assert len(results) > 0

    r2_vals = [r["R2"] for r in results]
    mean_r2 = float(np.nanmean(r2_vals))
    assert mean_r2 < 0.2, (
        f"Arm B with noise Δfeature should have R² ≈ 0, got mean R² = {mean_r2:.3f}"
    )

    mae_vals = [r["MAE"] for r in results]
    chance_vals = [r["chance_mae"] for r in results]
    mean_mae = float(np.nanmean(mae_vals))
    mean_chance = float(np.nanmean(chance_vals))
    assert mean_mae > mean_chance * 0.7, (
        f"Arm B noise model MAE ({mean_mae:.3f}) should be close to or above chance ({mean_chance:.3f})"
    )


# ---------------------------------------------------------------------------
# Positive control: Δfeature ∝ brain_change_rate → model recovers brain_change_rate
# ---------------------------------------------------------------------------

def test_arm_b_linear_signal_recovers_brain_change_rate():
    """When Δfeature = brain_change_rate * scale (plus tiny noise), the difference
    model should recover brain_change_rate well: R² > 0.5, MAE well below chance.
    """
    n_subjects = 100
    n_features = 20

    def linear_delta(bcr, delta_t, rng):
        # Feature change directly encodes the brain_change_rate scalar
        return np.full(n_features, bcr * 2.0) + rng.standard_normal(n_features) * 0.1

    manifest, subject_data = _make_cohort(
        n_subjects, n_features, linear_delta, seed=3,
        brain_change_rate_fn=lambda rng: float(rng.standard_normal()),
    )

    results = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation="difference",
        arm="B",
        band="all",
        n_outer_folds=5,
        n_inner_folds=3,
        ridge_alphas=[0.01, 0.1, 1.0, 10.0],
    )
    assert len(results) > 0

    r2_vals = [r["R2"] for r in results]
    mean_r2 = float(np.nanmean(r2_vals))
    assert mean_r2 > 0.5, (
        f"Arm B with linear brain_change_rate signal should have R² > 0.5, got {mean_r2:.3f}"
    )

    mae_vals = [r["MAE"] for r in results]
    chance_vals = [r["chance_mae"] for r in results]
    mean_mae = float(np.nanmean(mae_vals))
    mean_chance = float(np.nanmean(chance_vals))
    assert mean_mae < mean_chance * 0.7, (
        f"Arm B signal model MAE ({mean_mae:.3f}) should be well below chance ({mean_chance:.3f})"
    )


# ---------------------------------------------------------------------------
# Chance baseline: state inputs (mean) with brain_change_rate target → R² ≈ 0
# ---------------------------------------------------------------------------

def test_arm_b_mean_state_input_is_chance():
    """mean aggregation (state input) predicting brain_change_rate should score near chance.

    A model that sees only the average brain state cannot predict how fast the brain
    is changing — it has no access to temporal information.
    """
    n_subjects = 80
    n_features = 15

    def random_delta(bcr, delta_t, rng):
        return rng.standard_normal(n_features)

    manifest, subject_data = _make_cohort(n_subjects, n_features, random_delta, seed=5)

    results = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation="mean",
        arm="B",
        band="all",
        n_outer_folds=5,
        n_inner_folds=3,
        ridge_alphas=[0.1, 1.0, 10.0],
    )
    assert len(results) > 0

    r2_vals = [r["R2"] for r in results]
    mean_r2 = float(np.nanmean(r2_vals))
    assert mean_r2 < 0.15, (
        f"mean (state) input for brain_change_rate should have R² ≈ 0, got {mean_r2:.3f}"
    )


# ---------------------------------------------------------------------------
# Fold count
# ---------------------------------------------------------------------------

def test_correct_number_of_folds():
    """run_nested_cv should return exactly n_outer_folds results (or fewer if cohort too small)."""
    n_subjects = 50
    n_features = 10

    def noise(bcr, dt, rng):
        return rng.standard_normal(n_features)

    manifest, subject_data = _make_cohort(n_subjects, n_features, noise, seed=0)

    results = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation="mean",
        arm="A",
        band="all",
        n_outer_folds=5,
        n_inner_folds=3,
        ridge_alphas=[1.0],
    )
    assert len(results) == 5, f"Expected 5 folds, got {len(results)}"
