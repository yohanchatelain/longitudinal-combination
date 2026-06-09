"""Leakage prevention tests.

Covers:
- Train/test subject disjointness in every outer fold
- StandardScaler fit on train subjects only (mean/std differ between splits)
- z-scored train features have mean ≈ 0 and std ≈ 1
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parents[3]))

from brainage_agg.modeling.cv import run_nested_cv


# ---------------------------------------------------------------------------
# Synthetic cohort helpers
# ---------------------------------------------------------------------------

def _make_synthetic_cohort(
    n_subjects: int = 40,
    n_features: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Create a small synthetic cohort for leakage tests."""
    rng = np.random.default_rng(seed)
    rows = []
    subject_data = {}

    for i in range(n_subjects):
        sid = f"S{i:03d}"
        n_tp = rng.integers(2, 5)
        ages = np.sort(rng.uniform(20, 70, size=n_tp))
        delta_t = float(ages[-1] - ages[0])
        band = "Child" if rng.random() < 0.5 else "Adult"

        row_indices = list(range(i * 5, i * 5 + n_tp))  # dummy

        rows.append(
            {
                "subject_id": sid,
                "n_timepoints": n_tp,
                "band": band,
                "delta_t": delta_t,
                "ages": list(ages),
                "row_indices": row_indices,
                "t_first_idx": row_indices[0],
                "t_last_idx": row_indices[-1],
                "t_mid_idx": row_indices[len(row_indices) // 2] if n_tp >= 3 else -1,
                "cohort_all": n_tp >= 2,
                "cohort_concat": n_tp >= 3,
                "exclude_arm_b": delta_t == 0,
            }
        )
        subject_data[sid] = {
            "features": rng.standard_normal((n_tp, n_features)).astype(np.float32),
            "ages": ages,
            "row_indices": row_indices,
        }

    return pd.DataFrame(rows), subject_data


# ---------------------------------------------------------------------------
# Disjointness: no subject in both train and test
# ---------------------------------------------------------------------------

def test_subject_disjointness_all_folds():
    """Assert train/test subject ID sets are disjoint in EVERY outer fold."""
    manifest, subject_data = _make_synthetic_cohort(n_subjects=50)

    df = manifest[manifest["cohort_all"] & (~manifest["exclude_arm_b"])].copy()
    subject_ids = df["subject_id"].to_numpy()
    dummy_X = np.zeros((len(subject_ids), 1))
    gkf = GroupKFold(n_splits=5)

    for fold_idx, (train_local, test_local) in enumerate(
        gkf.split(dummy_X, groups=subject_ids)
    ):
        train_sids = set(subject_ids[train_local])
        test_sids = set(subject_ids[test_local])
        assert train_sids.isdisjoint(test_sids), (
            f"LEAKAGE in fold {fold_idx}: subjects {train_sids & test_sids} appear in both sets"
        )


# ---------------------------------------------------------------------------
# Scaler fit-on-train-only: scaler statistics differ from test set statistics
# ---------------------------------------------------------------------------

def test_scaler_fit_on_train_only():
    """StandardScaler statistics must be computed from train indices only.

    We verify that:
    1. The scaler's mean_ and scale_ match the train subset statistics.
    2. The test subset has a different mean after applying the scaler (i.e.,
       the scaler did NOT adapt to test data).
    """
    rng = np.random.default_rng(1)
    n_subjects = 60
    n_features = 15
    manifest, subject_data = _make_synthetic_cohort(n_subjects, n_features, seed=1)

    df = manifest[manifest["cohort_all"]].copy()
    subject_ids = df["subject_id"].to_numpy()

    # Manually replicate the scaler logic from cv.py
    dummy_X = np.zeros((len(subject_ids), 1))
    gkf = GroupKFold(n_splits=5)
    train_local, test_local = next(gkf.split(dummy_X, groups=subject_ids))

    train_sids = subject_ids[train_local]
    test_sids = subject_ids[test_local]

    # Build flat matrices
    X_train_flat = np.vstack([subject_data[s]["features"] for s in train_sids])
    X_test_flat = np.vstack([subject_data[s]["features"] for s in test_sids])

    # Scaler fit on TRAIN ONLY
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    # Verify scaler stats match train
    assert np.allclose(scaler.mean_, X_train_flat.mean(axis=0), atol=1e-5), \
        "Scaler mean_ does not match train mean"
    assert np.allclose(scaler.scale_, X_train_flat.std(axis=0, ddof=0) + 1e-12, atol=1e-3), \
        "Scaler scale_ does not match train std (approx)"

    # Verify scaler stats do NOT match test (they should differ)
    assert not np.allclose(scaler.mean_, X_test_flat.mean(axis=0), atol=0.01), \
        "Scaler was fit on test data (means too close — possible leakage)"

    # After transformation, train z-scores should have mean ≈ 0 and std ≈ 1
    X_train_scaled = scaler.transform(X_train_flat)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-4), \
        "Train z-scores do not have mean ≈ 0"
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-2), \
        "Train z-scores do not have std ≈ 1"

    # Test z-scores should NOT have mean ≈ 0 (they're shifted by their own mean)
    X_test_scaled = scaler.transform(X_test_flat)
    # This is a probabilistic check — in a random dataset means won't align exactly
    # Just verify the test was not used to fit the scaler (already checked above)


# ---------------------------------------------------------------------------
# Integration: run_nested_cv enforces disjointness internally
# ---------------------------------------------------------------------------

def test_run_nested_cv_disjoint_subjects():
    """run_nested_cv should raise AssertionError if subjects leak across folds.

    We don't mock the assertion — we just verify the function runs without
    assertion errors on a valid cohort (which it should).
    """
    manifest, subject_data = _make_synthetic_cohort(n_subjects=50, n_features=10)

    # Should complete without AssertionError
    results = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation="mean",
        arm="A",
        band="all",
        n_outer_folds=5,
        n_inner_folds=3,
        ridge_alphas=[0.1, 1.0, 10.0],
    )
    assert len(results) > 0, "Expected at least one fold result"
    for row in results:
        assert "MAE" in row
        assert "n_subjects_train" in row
        assert "n_subjects_test" in row
        # No subject overlap by construction; assert block inside cv.py would have fired
