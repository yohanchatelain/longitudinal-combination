"""Tests for aggregation functions.

Covers:
- annualized_rate is now valid in both Arm A and Arm B (Arm B target is brain_change_rate, not Δt)
- Anti-leakage: constant-offset shift leaves difference unchanged
- t_mid frozen: same seed → same t_mid; different seed → different t_mid
- Aggregation shapes match expectations
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))

from brainage_agg.agg.aggregations import (
    mean,
    concatenation,
    annualized_rate,
    difference,
    lme_slope,
    lme_slope_change,
    validate_arm,
    AGGREGATION_SPECS,
)
from brainage_agg.data.manifest import build_manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features():
    """3 timepoints × 10 features."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 10)).astype(np.float32)


@pytest.fixture
def synthetic_ages():
    return np.array([10.0, 11.5, 13.0])


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_mean_shape(synthetic_features):
    out = mean(synthetic_features)
    assert out.shape == (10,)


def test_concat_shape(synthetic_features):
    out = concatenation(synthetic_features, 0, 1, 2)
    assert out.shape == (30,)


def test_difference_shape(synthetic_features):
    out = difference(synthetic_features)
    assert out.shape == (10,)


def test_annualized_rate_shape(synthetic_features, synthetic_ages):
    out = annualized_rate(synthetic_features, synthetic_ages)
    assert out.shape == (10,)


# ---------------------------------------------------------------------------
# annualized_rate is valid in both arms (Arm B target is brain_change_rate, not Δt)
# ---------------------------------------------------------------------------

def test_annualized_rate_valid_in_arm_b(synthetic_features, synthetic_ages):
    """annualized_rate is now allowed in Arm B — no circularity with brain_change_rate target."""
    validate_arm("annualized_rate", "B")  # must not raise


def test_annualized_rate_valid_in_arm_a(synthetic_features, synthetic_ages):
    out = annualized_rate(synthetic_features, synthetic_ages)
    assert out.shape == (10,)


def test_validate_arm_blocks_difference_in_arm_a():
    with pytest.raises(ValueError):
        validate_arm("difference", "A")


def test_validate_arm_blocks_lme_slope_change_in_arm_a():
    with pytest.raises(ValueError):
        validate_arm("lme_slope_change", "A")


# ---------------------------------------------------------------------------
# Anti-leakage: constant offset does not affect difference
# ---------------------------------------------------------------------------

def test_difference_invariant_to_constant_offset(synthetic_features):
    """Adding a constant c to all timepoint features leaves difference unchanged.

    This is the key anti-leakage property for Arm B: the difference descriptor
    strips away the baseline state, so a subject with a higher absolute brain
    volume (offset) but identical trajectory looks the same.
    """
    c = np.full(10, 5.0)
    shifted = synthetic_features + c[None, :]  # add to all timepoints
    assert np.allclose(difference(synthetic_features), difference(shifted))


def test_difference_no_time_division(synthetic_features, synthetic_ages):
    """difference() should NOT divide by time — check against explicit delta."""
    expected = synthetic_features[-1] - synthetic_features[0]
    out = difference(synthetic_features)
    assert np.allclose(out, expected)

    # Also confirm it does NOT equal the annualized rate
    rate = annualized_rate(synthetic_features, synthetic_ages)
    delta_t = synthetic_ages[-1] - synthetic_ages[0]
    assert not np.allclose(out, rate), (
        "difference() should not equal annualized_rate — anti-circularity would be violated"
    )
    # Relationship: difference = rate * delta_t
    assert np.allclose(out, rate * delta_t, atol=1e-5)


# ---------------------------------------------------------------------------
# Annualized rate raises on zero age interval
# ---------------------------------------------------------------------------

def test_annualized_rate_zero_delta_raises():
    feats = np.ones((2, 5))
    ages = np.array([10.0, 10.0])  # same age
    with pytest.raises(ValueError, match="near zero"):
        annualized_rate(feats, ages)


# ---------------------------------------------------------------------------
# t_mid sampling reproducibility
# ---------------------------------------------------------------------------

def test_t_mid_frozen_same_seed(tmp_path):
    """Same seed → same t_mid_idx for every subject."""
    npz_path = Path(__file__).parents[3] / "outputs" / "features" / \
               "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
    if not npz_path.exists():
        pytest.skip("Feature file not available in test environment")

    m1 = build_manifest(npz_path, t_mid_seed=42)
    m2 = build_manifest(npz_path, t_mid_seed=42)
    concat_subs = m1[m1["cohort_concat"]]
    for _, row in concat_subs.iterrows():
        sid = row["subject_id"]
        idx1 = m1[m1["subject_id"] == sid]["t_mid_idx"].values[0]
        idx2 = m2[m2["subject_id"] == sid]["t_mid_idx"].values[0]
        assert idx1 == idx2, f"t_mid_idx differs for subject {sid} across builds with same seed"


def test_t_mid_varies_with_different_seed(tmp_path):
    """Different seeds produce at least some different t_mid assignments."""
    npz_path = Path(__file__).parents[3] / "outputs" / "features" / \
               "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
    if not npz_path.exists():
        pytest.skip("Feature file not available in test environment")

    m1 = build_manifest(npz_path, t_mid_seed=42)
    m2 = build_manifest(npz_path, t_mid_seed=99)
    concat_subs = m1[m1["cohort_concat"] & (m1["n_timepoints"] >= 4)]  # need ≥4 tp for t_mid to vary
    if len(concat_subs) == 0:
        pytest.skip("No subjects with ≥4 timepoints in this dataset")
    mids_1 = m1[m1["subject_id"].isin(concat_subs["subject_id"])]["t_mid_idx"].values
    mids_2 = m2[m2["subject_id"].isin(concat_subs["subject_id"])]["t_mid_idx"].values
    assert not np.array_equal(mids_1, mids_2), "Different seeds should produce different t_mid assignments"


# ---------------------------------------------------------------------------
# LME slope lookup
# ---------------------------------------------------------------------------

def test_lme_slope_lookup():
    subjects = np.array(["S1", "S2", "S3"])
    slopes = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = lme_slope("S2", subjects, slopes)
    assert np.allclose(out, [3.0, 4.0])


def test_lme_slope_missing_raises():
    subjects = np.array(["S1", "S2"])
    slopes = np.zeros((2, 5))
    with pytest.raises(KeyError):
        lme_slope("S_UNKNOWN", subjects, slopes)


# ---------------------------------------------------------------------------
# AGGREGATION_SPECS completeness
# ---------------------------------------------------------------------------

def test_all_arm_a_aggregations_registered():
    for agg in ["mean", "concatenation", "annualized_rate", "lme_slope"]:
        assert "A" in AGGREGATION_SPECS[agg]["arms"], f"{agg} not registered for Arm A"


def test_all_arm_b_aggregations_registered():
    for agg in ["annualized_rate", "difference", "lme_slope_change", "mean", "concatenation"]:
        assert "B" in AGGREGATION_SPECS[agg]["arms"], f"{agg} not registered for Arm B"
