"""CPU-only tests for the geometric longitudinal validation pipeline.

See geometric_longitudinal_validation_architecture.md for the design and
geometric_longitudinal_validation_plan.md for the scientific protocol. This file
grows alongside the geometric/ package build order in the architecture doc §7.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy import ndimage as ndi

from brainage_agg.experiment.run_geometric_validation import (
    _channels_for_variant,
    _planned_cells,
    deterministic_pilot_validation_split,
    prepare,
    process_condition,
    resolve_sigma_voxels,
    target_relevant_conditions,
)
from brainage_agg.geometric.config import load_geometric_config, validate_locked_config
from brainage_agg.geometric.decoder import (
    bootstrap_prediction_intervals,
    fit_decoder,
    predict as decoder_predict,
    select_ridge_alpha,
)
from brainage_agg.geometric.deformation import (
    _index_grid,
    calibrate_magnitude_for_target_change,
    global_scaling_field,
    identity_field,
    jacobian_determinant,
    radial_bump_field,
    realized_volume_change_pct,
    rigid_subvoxel_field,
)
from brainage_agg.geometric.determinism import (
    assert_float32,
    enforce_deterministic_environment,
    preprocess_pair,
    reproducibility_self_test,
)
from brainage_agg.geometric.invariance_layer import (
    apply_invariance_layer,
    band_limit_filter,
    calibrate_filter_cutoff,
    preprocess_pair_with_variant,
)
from brainage_agg.geometric.qc import evaluate_deformation_qc
from brainage_agg.geometric.self_calibration import (
    compute_null_response_distribution,
    compute_self_calibration,
    null_resample_fields,
    to_dataframe as self_calibration_to_dataframe,
    zscore_against_null,
)
from brainage_agg.geometric.synthesis import (
    invert_displacement_field,
    synthesize_followup,
    warp_image,
)
from brainage_agg.geometric.frozen_cnn import (
    build_frozen_cnn,
    build_response_fn,
    extract_features,
    feature_change_magnitude,
)
from brainage_agg.geometric.gates import (
    check_bootstrap_excludes_zero,
    check_calibration_gate,
    check_consistent_direction_across_cohorts,
    check_error_difference_favors_method,
    check_false_positive_rate_no_worse,
    check_invariant_variant_justified,
    check_not_single_seed_driven,
    check_practically_meaningful_improvement,
    evaluate_superiority_claim,
)
from brainage_agg.geometric.statistics import (
    calibration_slope_intercept,
    equivalence_test_tost,
    interval_coverage,
    paired_change_test,
    power_curve_by_resampling,
    regression_accuracy_metrics,
    sign_accuracy,
    spearman_dose_response,
    subject_clustered_bootstrap_ci,
    subject_level_discrimination_auroc,
)
from brainage_agg.geometric.trained_cnn import (
    TrainingExample,
    assert_no_training_subject_overlap,
    build_cnn,
    load_trained_checkpoint,
    save_checkpoint,
    train_cnn,
)


ROOT = Path(__file__).resolve().parents[2]


def _sphere_mask(shape: tuple[int, int, int], center: tuple[int, int, int], radius: float) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = center
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def test_assert_float32_accepts_float32_and_rejects_other_dtypes():
    assert_float32(np.zeros(3, dtype=np.float32), np.ones(2, dtype=np.float32))
    with pytest.raises(TypeError, match="float64"):
        assert_float32(np.zeros(3, dtype=np.float64))


def test_preprocess_pair_uses_one_call_site_with_shared_kwargs():
    mask = np.ones((6, 6, 6), dtype=bool)
    baseline = np.random.default_rng(0).uniform(0, 100, size=(6, 6, 6)).astype(np.float32)
    followup = baseline.copy()  # exact duplicate: zero-change condition

    baseline_channels, followup_channels = preprocess_pair(
        baseline, followup, mask, scaler="minmax", channels=["t1", "sobel"],
    )
    np.testing.assert_array_equal(baseline_channels, followup_channels)
    assert baseline_channels.shape[0] == 2


def test_preprocess_pair_rejects_non_float32_input():
    mask = np.ones((4, 4, 4), dtype=bool)
    baseline = np.zeros((4, 4, 4), dtype=np.float64)
    followup = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="preprocess_pair input"):
        preprocess_pair(baseline, followup, mask, scaler="minmax", channels=["t1"])


def test_reproducibility_self_test_passes_for_a_deterministic_pipeline():
    rng_seed = 7

    def deterministic_pipeline() -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        return rng.standard_normal(1000)

    result = reproducibility_self_test(deterministic_pipeline, tolerance=1e-12, seed=rng_seed)
    assert result["passed"]
    assert result["max_abs_diff"] == 0.0
    assert result["n_values"] == 1000


def test_reproducibility_self_test_fails_for_a_non_deterministic_pipeline():
    state = {"call": 0}

    def flaky_pipeline() -> np.ndarray:
        state["call"] += 1
        return np.random.default_rng(state["call"]).standard_normal(100)

    result = reproducibility_self_test(flaky_pipeline, tolerance=1e-9, seed=1)
    assert not result["passed"]
    assert result["max_abs_diff"] > 0.0


def test_reproducibility_self_test_rejects_mismatched_output_shape():
    state = {"call": 0}

    def shape_drift_pipeline() -> np.ndarray:
        state["call"] += 1
        return np.zeros(state["call"])

    with pytest.raises(ValueError, match="output shape"):
        reproducibility_self_test(shape_drift_pipeline, tolerance=1e-9, seed=1)


def test_enforce_deterministic_environment_is_idempotent_and_seeds_numpy():
    enforce_deterministic_environment(seed=42)
    first = np.random.standard_normal(5)
    enforce_deterministic_environment(seed=42)
    second = np.random.standard_normal(5)
    np.testing.assert_array_equal(first, second)


# --- deformation.py ---------------------------------------------------------------

SHAPE = (21, 21, 21)
CENTER = (10, 10, 10)


def test_identity_field_has_zero_ground_truth_change():
    field = identity_field(SHAPE)
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    jacobian = jacobian_determinant(field)
    np.testing.assert_allclose(jacobian, 1.0, atol=1e-6)
    assert realized_volume_change_pct(jacobian, mask) == pytest.approx(0.0, abs=1e-4)


def test_rigid_subvoxel_field_has_exactly_zero_ground_truth_change():
    field = rigid_subvoxel_field(
        SHAPE, translation_voxels=(0.3, -0.2, 0.1), rotation_degrees=(1.0, 0.5, -0.7),
    )
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    jacobian = jacobian_determinant(field)
    # A rigid transform's Jacobian is exactly the (constant) rotation matrix, so
    # det == 1 everywhere up to floating-point roundoff -- no finite-difference
    # truncation error, since the field is affine in x.
    np.testing.assert_allclose(jacobian, 1.0, atol=1e-4)
    assert realized_volume_change_pct(jacobian, mask) == pytest.approx(0.0, abs=1e-2)


def test_global_scaling_field_matches_closed_form_jacobian():
    scale = 0.9
    field = global_scaling_field(SHAPE, scale=scale)
    jacobian = jacobian_determinant(field)
    np.testing.assert_allclose(jacobian, scale**3, atol=1e-4)
    mask = np.ones(SHAPE, dtype=bool)
    expected_pct = 100.0 * (scale**3 - 1.0)
    assert realized_volume_change_pct(jacobian, mask) == pytest.approx(expected_pct, abs=1e-2)


def test_global_scaling_field_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale must be positive"):
        global_scaling_field(SHAPE, scale=0.0)


@pytest.mark.parametrize("direction,expected_sign", [("contract", -1), ("expand", 1)])
def test_radial_bump_field_direction_controls_sign_of_change(direction, expected_sign):
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    field = radial_bump_field(SHAPE, mask, magnitude=0.1, sigma_voxels=4.0, direction=direction)
    jacobian = jacobian_determinant(field)
    realized = realized_volume_change_pct(jacobian, mask)
    assert np.sign(realized) == expected_sign


def test_radial_bump_field_magnitude_zero_is_a_no_op():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    field = radial_bump_field(SHAPE, mask, magnitude=0.0, sigma_voxels=4.0, direction="contract")
    jacobian = jacobian_determinant(field)
    assert realized_volume_change_pct(jacobian, mask) == pytest.approx(0.0, abs=1e-6)


def test_radial_bump_field_effect_grows_monotonically_with_magnitude():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)

    def realized(magnitude: float) -> float:
        field = radial_bump_field(SHAPE, mask, magnitude=magnitude, sigma_voxels=4.0, direction="contract")
        return realized_volume_change_pct(jacobian_determinant(field), mask)

    values = [realized(m) for m in (0.0, 0.1, 0.2, 0.3)]
    assert values == sorted(values, reverse=True)  # increasingly negative (more contraction)


def test_radial_bump_field_rejects_negative_magnitude():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    with pytest.raises(ValueError, match="non-negative"):
        radial_bump_field(SHAPE, mask, magnitude=-0.1, sigma_voxels=4.0, direction="contract")


def test_radial_bump_field_rejects_unknown_direction():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    with pytest.raises(ValueError, match="direction"):
        radial_bump_field(SHAPE, mask, magnitude=0.1, sigma_voxels=4.0, direction="sideways")


def test_calibrate_magnitude_for_target_change_hits_the_realized_target():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    builder = lambda magnitude: radial_bump_field(  # noqa: E731
        SHAPE, mask, magnitude=magnitude, sigma_voxels=4.0, direction="contract",
    )
    magnitude, field, realized = calibrate_magnitude_for_target_change(
        builder, mask, target_change_pct=-5.0, magnitude_bounds=(0.0, 0.1), tolerance_pct=0.05,
    )
    assert realized == pytest.approx(-5.0, abs=0.05)
    assert magnitude > 0.0
    assert field.condition == "radial_contract"


def test_calibrate_magnitude_for_target_change_rejects_unreachable_target():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    builder = lambda magnitude: radial_bump_field(  # noqa: E731
        SHAPE, mask, magnitude=magnitude, sigma_voxels=4.0, direction="contract",
    )
    with pytest.raises(ValueError, match="outside the achievable range"):
        calibrate_magnitude_for_target_change(
            builder, mask, target_change_pct=-99.0, magnitude_bounds=(0.0, 0.05), tolerance_pct=0.05,
        )


# --- qc.py -------------------------------------------------------------------------

def test_evaluate_deformation_qc_passes_a_well_calibrated_field():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    transition_band = ndi.binary_dilation(mask, iterations=3)
    _, field, realized = calibrate_magnitude_for_target_change(
        lambda magnitude: radial_bump_field(SHAPE, mask, magnitude=magnitude, sigma_voxels=4.0, direction="contract"),
        mask, target_change_pct=-5.0, magnitude_bounds=(0.0, 0.1), tolerance_pct=0.05,
    )
    result = evaluate_deformation_qc(
        field,
        region_mask=mask,
        transition_band_mask=transition_band,
        requested_change_pct=-5.0,
        tolerance_pct=0.2,
        max_outside_change_pct=5.0,
        max_boundary_gradient=1.0,
        baseline_shape=SHAPE,
        followup_shape=SHAPE,
    )
    assert result.passed, result.checks
    assert result.checks["positive_jacobian"]
    assert result.values["realized_change_pct"] == pytest.approx(realized, abs=1e-6)


def test_evaluate_deformation_qc_fails_on_folding_transformation():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    transition_band = ndi.binary_dilation(mask, iterations=3)
    field = radial_bump_field(SHAPE, mask, magnitude=1.5, sigma_voxels=4.0, direction="contract")
    result = evaluate_deformation_qc(
        field,
        region_mask=mask,
        transition_band_mask=transition_band,
        requested_change_pct=-90.0,
        tolerance_pct=5.0,
        max_outside_change_pct=5.0,
        max_boundary_gradient=1.0,
        baseline_shape=SHAPE,
        followup_shape=SHAPE,
    )
    assert not result.passed
    assert not result.checks["positive_jacobian"]



# --- synthesis.py --------------------------------------------------------------------

def _sphere_image() -> np.ndarray:
    return _sphere_mask(SHAPE, CENTER, radius=6).astype(np.float32)


def test_warp_image_with_identity_field_is_exact():
    image = _sphere_image()
    field = identity_field(SHAPE)
    warped = warp_image(image, field)
    np.testing.assert_array_equal(warped, image)


def test_warp_image_with_zero_rigid_field_is_exact():
    image = _sphere_image()
    field = rigid_subvoxel_field(SHAPE, translation_voxels=(0, 0, 0), rotation_degrees=(0, 0, 0))
    warped = warp_image(image, field)
    np.testing.assert_array_equal(warped, image)


def test_warp_image_with_global_scaling_matches_expected_volume_change():
    image = _sphere_image()
    baseline_voxels = float(image.sum())

    shrunk = warp_image(image, global_scaling_field(SHAPE, scale=0.5))
    grown = warp_image(image, global_scaling_field(SHAPE, scale=1.5))

    shrunk_voxels = float((shrunk > 0.5).sum())
    grown_voxels = float((grown > 0.5).sum())
    assert shrunk_voxels == pytest.approx(baseline_voxels * 0.5**3, rel=0.1)
    assert grown_voxels == pytest.approx(baseline_voxels * 1.5**3, rel=0.1)


def test_warp_image_rejects_shape_mismatch():
    image = np.zeros((5, 5, 5), dtype=np.float32)
    field = identity_field(SHAPE)
    with pytest.raises(ValueError, match="shape"):
        warp_image(image, field)


def test_invert_displacement_field_is_accurate_away_from_the_grid_boundary():
    field = rigid_subvoxel_field(
        SHAPE, translation_voxels=(0.5, -0.3, 0.2), rotation_degrees=(5.0, 3.0, -4.0),
    )
    inverse = invert_displacement_field(field.displacement, iterations=12)
    grid = _index_grid(SHAPE)
    estimated_baseline_point = grid + inverse
    forward_displacement = np.stack(
        [ndi.map_coordinates(field.displacement[c], estimated_baseline_point, order=1, mode="nearest")
         for c in range(3)]
    )
    reconstructed = estimated_baseline_point + forward_displacement
    central = _sphere_mask(SHAPE, CENTER, radius=8)
    error = np.linalg.norm(reconstructed - grid, axis=0)
    assert float(error[central].max()) < 1e-3


def test_synthesize_followup_returns_grid_identical_image_and_mask():
    image = _sphere_image()
    mask = image > 0.5
    field = radial_bump_field(SHAPE, mask, magnitude=0.3, sigma_voxels=4.0, direction="contract")
    followup_image, followup_mask = synthesize_followup(image, mask, field)
    assert followup_image.shape == image.shape
    assert followup_mask.shape == mask.shape
    assert followup_mask.dtype == bool
    # contraction: fewer foreground voxels in the follow-up than the baseline
    assert followup_mask.sum() < mask.sum()



# --- invariance_layer.py --------------------------------------------------------------

def test_band_limit_filter_zero_sigma_is_identity_inside_mask():
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    volume = (np.random.default_rng(0).uniform(0, 100, size=SHAPE)).astype(np.float32)
    filtered = band_limit_filter(volume, mask, sigma_voxels=0.0)
    np.testing.assert_array_equal(filtered[mask], volume[mask])
    assert (filtered[~mask] == 0.0).all()


def test_band_limit_filter_is_unbiased_for_uniform_intensity_near_boundary():
    # A normalized (mask-aware) filter must reproduce a constant value everywhere
    # inside the mask, including right at the boundary -- a naive
    # gaussian_filter(volume * mask) would pull boundary voxels toward zero instead.
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    constant_value = 50.0
    volume = (mask.astype(np.float32) * constant_value)
    filtered = band_limit_filter(volume, mask, sigma_voxels=2.0)
    np.testing.assert_allclose(filtered[mask], constant_value, atol=1e-3)
    assert (filtered[~mask] == 0.0).all()


def test_band_limit_filter_reduces_high_frequency_variance():
    mask = np.ones(SHAPE, dtype=bool)
    rng = np.random.default_rng(1)
    volume = rng.standard_normal(SHAPE).astype(np.float32) * 10.0 + 50.0
    filtered = band_limit_filter(volume, mask, sigma_voxels=2.0)
    assert filtered[mask].std() < volume[mask].std()


def test_band_limit_filter_rejects_negative_sigma():
    mask = np.ones(SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="non-negative"):
        band_limit_filter(np.zeros(SHAPE, dtype=np.float32), mask, sigma_voxels=-1.0)


def test_apply_invariance_layer_raw_variant_is_identity():
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    volume = np.random.default_rng(2).uniform(0, 100, size=SHAPE).astype(np.float32)
    out = apply_invariance_layer(volume, mask, variant="raw", sigma_voxels=3.0)
    np.testing.assert_array_equal(out, volume)


def test_apply_invariance_layer_invariant_variant_matches_band_limit_filter():
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    volume = np.random.default_rng(3).uniform(0, 100, size=SHAPE).astype(np.float32)
    out = apply_invariance_layer(volume, mask, variant="invariant", sigma_voxels=2.0)
    expected = band_limit_filter(volume, mask, sigma_voxels=2.0)
    np.testing.assert_array_equal(out, expected)


def test_apply_invariance_layer_rejects_unknown_variant():
    mask = np.ones(SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="Unknown variant"):
        apply_invariance_layer(np.zeros(SHAPE, dtype=np.float32), mask, variant="fancy", sigma_voxels=1.0)


def test_preprocess_pair_with_variant_treats_baseline_and_followup_symmetrically():
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    baseline = np.random.default_rng(4).uniform(0, 100, size=SHAPE).astype(np.float32)
    followup = baseline.copy()
    baseline_channels, followup_channels = preprocess_pair_with_variant(
        baseline, followup, mask,
        variant="invariant", sigma_voxels=2.0, scaler="minmax", channels=["t1", "sobel"],
    )
    np.testing.assert_array_equal(baseline_channels, followup_channels)


def test_preprocess_pair_with_variant_raw_matches_plain_preprocess_pair():
    mask = _sphere_mask(SHAPE, CENTER, radius=6)
    baseline = np.random.default_rng(5).uniform(0, 100, size=SHAPE).astype(np.float32)
    followup = np.random.default_rng(6).uniform(0, 100, size=SHAPE).astype(np.float32)
    from brainage_agg.geometric.determinism import preprocess_pair as plain_preprocess_pair

    variant_baseline, variant_followup = preprocess_pair_with_variant(
        baseline, followup, mask, variant="raw", sigma_voxels=2.0, scaler="minmax", channels=["t1"],
    )
    plain_baseline, plain_followup = plain_preprocess_pair(
        baseline, followup, mask, scaler="minmax", channels=["t1"],
    )
    np.testing.assert_array_equal(variant_baseline, plain_baseline)
    np.testing.assert_array_equal(variant_followup, plain_followup)


def test_calibrate_filter_cutoff_picks_lowest_sham_response_meeting_retention():
    sham_response = {0.0: np.array([1.0, 1.2]), 1.0: np.array([0.5, 0.6]), 2.0: np.array([0.1, 0.15])}
    dose_response = {0.0: np.array([10.0, 11.0]), 1.0: np.array([9.0, 9.5]), 2.0: np.array([5.0, 5.2])}
    reference = dose_response[0.0]
    result = calibrate_filter_cutoff(
        [0.0, 1.0, 2.0], sham_response, dose_response,
        reference_dose_response=reference, min_dose_retention_fraction=0.8,
    )
    # sigma=2.0 has the lowest sham response but fails the 80% retention constraint
    # (dose mean 5.1 / 10.5 ~= 0.49), so sigma=1.0 (retention ~0.88) should win.
    assert result.sigma_voxels == pytest.approx(1.0)
    assert len(result.table) == 3


def test_calibrate_filter_cutoff_raises_when_no_candidate_meets_retention():
    sham_response = {0.5: np.array([0.2]), 1.0: np.array([0.1])}
    dose_response = {0.5: np.array([1.0]), 1.0: np.array([0.5])}
    with pytest.raises(ValueError, match="retains at least"):
        calibrate_filter_cutoff(
            [0.5, 1.0], sham_response, dose_response,
            reference_dose_response=np.array([10.0]), min_dose_retention_fraction=0.8,
        )


def test_calibrate_filter_cutoff_raises_on_missing_candidate_response():
    with pytest.raises(ValueError, match="Missing sham/dose response"):
        calibrate_filter_cutoff(
            [0.0, 1.0], {0.0: np.array([1.0])}, {0.0: np.array([10.0])},
            reference_dose_response=np.array([10.0]),
        )


def test_evaluate_deformation_qc_fails_on_grid_mismatch():
    mask = _sphere_mask(SHAPE, CENTER, radius=5)
    field = identity_field(SHAPE)
    result = evaluate_deformation_qc(
        field,
        region_mask=mask,
        transition_band_mask=mask,
        requested_change_pct=0.0,
        tolerance_pct=0.5,
        max_outside_change_pct=5.0,
        max_boundary_gradient=1.0,
        baseline_shape=SHAPE,
        followup_shape=(20, 21, 21),
    )
    assert not result.passed
    assert not result.checks["grid_identical"]


# --- self_calibration.py ------------------------------------------------------------

def test_null_resample_fields_are_reproducible_and_within_bounds():
    first = null_resample_fields(SHAPE, n_resamples=5, seed=11, max_translation_voxels=0.5, max_rotation_degrees=1.0)
    second = null_resample_fields(SHAPE, n_resamples=5, seed=11, max_translation_voxels=0.5, max_rotation_degrees=1.0)
    assert len(first) == 5
    for a, b in zip(first, second):
        assert a.condition == "resampling_sham" == b.condition
        assert a.params == b.params
        assert max(abs(v) for v in a.params["translation_voxels"]) <= 0.5
        assert max(abs(v) for v in a.params["rotation_degrees"]) <= 1.0


def test_null_resample_fields_rejects_too_few_resamples():
    with pytest.raises(ValueError, match="n_resamples"):
        null_resample_fields(SHAPE, n_resamples=0, seed=1)


def test_compute_self_calibration_zscores_against_own_null_distribution():
    baseline = _sphere_image()
    mask = baseline > 0.5

    def response_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).sum())

    result = compute_self_calibration(
        "sub-001",
        observed_response=1000.0,  # deliberately far from the null distribution
        baseline_image=baseline,
        baseline_mask=mask,
        response_fn=response_fn,
        n_resamples=5,
        seed=3,
    )
    assert result.subject_id == "sub-001"
    assert len(result.null_responses) == 5
    assert result.null_std >= 0.0
    assert result.z_score == pytest.approx(
        (1000.0 - result.null_mean) / max(result.null_std, 1e-6)
    )
    assert result.z_score > 0  # observed response is far above the null


def test_compute_self_calibration_is_reproducible_for_a_fixed_seed():
    baseline = _sphere_image()
    mask = baseline > 0.5

    def response_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean())

    first = compute_self_calibration(
        "sub-001", observed_response=5.0, baseline_image=baseline, baseline_mask=mask,
        response_fn=response_fn, n_resamples=4, seed=7,
    )
    second = compute_self_calibration(
        "sub-001", observed_response=5.0, baseline_image=baseline, baseline_mask=mask,
        response_fn=response_fn, n_resamples=4, seed=7,
    )
    np.testing.assert_array_equal(first.null_responses, second.null_responses)
    assert first.z_score == second.z_score


def test_compute_self_calibration_floors_the_denominator_for_degenerate_null():
    baseline = _sphere_image()
    mask = baseline > 0.5
    result = compute_self_calibration(
        "sub-001", observed_response=5.0, baseline_image=baseline, baseline_mask=mask,
        response_fn=lambda a, b: 2.0,  # constant response -> zero null variance
        n_resamples=3, seed=1, min_null_std=1e-3,
    )
    assert result.null_std == 0.0
    assert result.z_score == pytest.approx((5.0 - 2.0) / 1e-3)
    assert np.isfinite(result.z_score)


def test_compute_self_calibration_rejects_too_few_resamples():
    baseline = _sphere_image()
    mask = baseline > 0.5
    with pytest.raises(ValueError, match="n_resamples"):
        compute_self_calibration(
            "sub-001", observed_response=1.0, baseline_image=baseline, baseline_mask=mask,
            response_fn=lambda a, b: 0.0, n_resamples=1, seed=1,
        )


def test_self_calibration_to_dataframe_flattens_results():
    baseline = _sphere_image()
    mask = baseline > 0.5
    results = [
        compute_self_calibration(
            f"sub-{index:03d}", observed_response=float(index), baseline_image=baseline,
            baseline_mask=mask, response_fn=lambda a, b: float(np.abs(a - b).mean()),
            n_resamples=3, seed=index,
        )
        for index in range(3)
    ]
    frame = self_calibration_to_dataframe(results)
    assert list(frame["subject_id"]) == ["sub-000", "sub-001", "sub-002"]
    assert (frame["n_null_resamples"] == 3).all()
    assert {"observed_response", "null_mean", "null_std", "z_score"}.issubset(frame.columns)


def test_compute_null_response_distribution_matches_compute_self_calibration():
    baseline = _sphere_image()
    mask = baseline > 0.5

    def response_fn(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean())

    null_responses = compute_null_response_distribution(
        baseline, mask, response_fn, n_resamples=5, seed=9,
    )
    result = compute_self_calibration(
        "sub-001", observed_response=1.0, baseline_image=baseline, baseline_mask=mask,
        response_fn=response_fn, n_resamples=5, seed=9,
    )
    np.testing.assert_array_equal(null_responses, result.null_responses)


def test_zscore_against_null_multiple_observations_share_one_null():
    null_responses = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    scored_low = zscore_against_null(1.0, null_responses)
    scored_high = zscore_against_null(5.0, null_responses)
    assert scored_low["null_mean"] == scored_high["null_mean"]
    assert scored_low["null_std"] == scored_high["null_std"]
    assert scored_high["z_score"] > scored_low["z_score"]


def test_zscore_against_null_rejects_too_few_null_values():
    with pytest.raises(ValueError, match="at least two"):
        zscore_against_null(1.0, np.array([1.0]))


# --- trained_cnn.py -------------------------------------------------------------------
# Volumes must be at least 32^3 to survive the backbone's 4 AvgPool3d(2) stages plus
# its AdaptiveAvgPool3d(2) head; 6-8 examples and 2-3 epochs keep these CPU-only.

def _training_examples(prefix: str, n: int, *, seed: int) -> list[TrainingExample]:
    rng = np.random.default_rng(seed)
    return [
        TrainingExample(
            subject_id=f"{prefix}-{index}",
            channels=rng.uniform(0, 1, size=(2, 32, 32, 32)).astype(np.float32),
            label=float(20 + index),
        )
        for index in range(n)
    ]


def test_build_cnn_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown arch"):
        build_cnn("resnet50", in_channels=2)


def test_assert_no_training_subject_overlap_raises_on_shared_ids():
    with pytest.raises(ValueError, match="appear in both"):
        assert_no_training_subject_overlap(["a", "b"], ["b", "c"])


def test_assert_no_training_subject_overlap_passes_when_disjoint():
    assert_no_training_subject_overlap(["a", "b"], ["c", "d"])  # no raise


def test_train_cnn_rejects_train_val_subject_overlap():
    train_ex = _training_examples("sub", 4, seed=0)
    val_ex = [train_ex[0]]  # same subject_id as a training example
    with pytest.raises(ValueError, match="appear in both"):
        train_cnn("double_conv", train_ex, val_ex, seed=0, max_epochs=1, batch_size=2)


def test_train_cnn_rejects_held_out_subject_in_training_set():
    train_ex = _training_examples("sub", 4, seed=0)
    val_ex = _training_examples("val", 2, seed=1)
    with pytest.raises(ValueError, match="held-out"):
        train_cnn(
            "double_conv", train_ex, val_ex,
            held_out_subject_ids=["sub-0"], seed=0, max_epochs=1, batch_size=2,
        )


def test_train_cnn_runs_and_returns_a_best_checkpoint():
    train_ex = _training_examples("train", 6, seed=0)
    val_ex = _training_examples("val", 3, seed=1)
    result = train_cnn(
        "double_conv", train_ex, val_ex,
        held_out_subject_ids=["geo-pilot-1"], seed=0, max_epochs=3, patience=3, batch_size=2,
    )
    assert result.architecture == "double_conv"
    assert result.in_channels == 2
    assert result.best_epoch >= 0
    assert np.isfinite(result.best_val_loss)
    assert len(result.history) >= 1
    assert result.backbone_state_dict  # non-empty


def test_train_cnn_is_reproducible_for_a_fixed_seed():
    train_ex = _training_examples("train", 6, seed=0)
    val_ex = _training_examples("val", 3, seed=1)
    first = train_cnn("double_conv", train_ex, val_ex, seed=0, max_epochs=2, patience=2, batch_size=2)
    second = train_cnn("double_conv", train_ex, val_ex, seed=0, max_epochs=2, patience=2, batch_size=2)
    assert first.best_val_loss == second.best_val_loss
    for key in first.backbone_state_dict:
        assert torch.equal(first.backbone_state_dict[key], second.backbone_state_dict[key])


def test_save_and_load_checkpoint_round_trips_to_a_frozen_backbone(tmp_path):
    train_ex = _training_examples("train", 6, seed=0)
    val_ex = _training_examples("val", 3, seed=1)
    result = train_cnn("double_conv", train_ex, val_ex, seed=0, max_epochs=2, patience=2, batch_size=2)

    checkpoint_path = save_checkpoint(result, tmp_path)
    assert checkpoint_path.is_file()

    backbone = load_trained_checkpoint(checkpoint_path, device="cpu")
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())

    example_input = torch.as_tensor(train_ex[0].channels[None], dtype=torch.float32)
    with torch.no_grad():
        first_output = backbone(example_input)
        second_output = backbone(example_input)
    torch.testing.assert_close(first_output, second_output)


# --- statistics.py --------------------------------------------------------------------

def test_paired_change_test_detects_a_clear_positive_shift():
    differences = np.array([2.0, 2.5, 1.8, 2.2, 1.9, 2.1])
    result = paired_change_test(differences)
    assert result["n"] == 6
    assert result["mean_difference"] == pytest.approx(2.083, abs=0.01)
    assert result["t_p_value"] < 0.01
    assert result["wilcoxon_p_value"] < 0.10


def test_paired_change_test_rejects_too_few_values():
    with pytest.raises(ValueError, match="at least two"):
        paired_change_test(np.array([1.0]))


def test_subject_clustered_bootstrap_ci_is_reproducible_and_excludes_zero_for_clear_effect():
    subject_ids = [f"s{i}" for i in range(10) for _ in range(2)]
    values = np.array([5.0 + 0.1 * i for i in range(10) for _ in range(2)])
    first = subject_clustered_bootstrap_ci(values, subject_ids, np.mean, n_bootstrap=500, seed=1)
    second = subject_clustered_bootstrap_ci(values, subject_ids, np.mean, n_bootstrap=500, seed=1)
    assert first == second
    assert first["excludes_zero"]
    assert first["n_subjects"] == 10


def test_subject_clustered_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        subject_clustered_bootstrap_ci([1.0, 2.0], ["a"], np.mean, seed=1)


def test_subject_clustered_bootstrap_ci_rejects_single_subject():
    with pytest.raises(ValueError, match="at least two subjects"):
        subject_clustered_bootstrap_ci([1.0, 2.0], ["a", "a"], np.mean, seed=1)


def test_spearman_dose_response_is_perfect_for_a_monotonic_relationship():
    dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    response = dose * 2.0 + 1.0
    result = spearman_dose_response(dose, response)
    assert result["spearman_r"] == pytest.approx(1.0)
    assert result["n"] == 5


def test_spearman_dose_response_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least three"):
        spearman_dose_response([1.0, 2.0], [1.0, 2.0])


def test_calibration_slope_intercept_recovers_a_perfect_line():
    true_values = np.array([1.0, 2.0, 3.0, 4.0])
    predicted_values = true_values.copy()
    result = calibration_slope_intercept(true_values, predicted_values)
    assert result["slope"] == pytest.approx(1.0, abs=1e-8)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-8)


def test_sign_accuracy_counts_matching_signs_and_excludes_zeros():
    true_values = np.array([1.0, -1.0, 2.0, 0.0])
    predicted_values = np.array([1.0, -1.0, -2.0, 5.0])
    # index 3 (true=0) is excluded; of the remaining 3, 2 match sign
    assert sign_accuracy(true_values, predicted_values) == pytest.approx(2 / 3)


def test_sign_accuracy_rejects_when_no_defined_sign_exists():
    with pytest.raises(ValueError, match="defined sign"):
        sign_accuracy(np.array([0.0, 0.0]), np.array([1.0, -1.0]))


def test_subject_level_discrimination_auroc_is_perfect_for_separable_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    assert subject_level_discrimination_auroc(labels, scores) == pytest.approx(1.0)


def test_subject_level_discrimination_auroc_rejects_a_single_class():
    with pytest.raises(ValueError, match="both classes"):
        subject_level_discrimination_auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))


def test_regression_accuracy_metrics_computes_known_errors():
    true_values = np.array([10.0, 20.0, 30.0])
    predicted_values = np.array([12.0, 18.0, 33.0])
    # errors: +2, -2, +3 -> mae=7/3, rmse=sqrt((4+4+9)/3), bias=+1
    result = regression_accuracy_metrics(true_values, predicted_values)
    assert result["mae"] == pytest.approx(7 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(17 / 3))
    assert result["signed_bias"] == pytest.approx(1.0)


def test_interval_coverage_counts_true_values_inside_bounds():
    true_values = np.array([1.0, 5.0, 10.0])
    lower_bounds = np.array([0.0, 6.0, 8.0])
    upper_bounds = np.array([2.0, 7.0, 12.0])
    # covered: index0 (1 in [0,2]) yes, index1 (5 in [6,7]) no, index2 (10 in [8,12]) yes
    assert interval_coverage(true_values, lower_bounds, upper_bounds) == pytest.approx(2 / 3)


def test_equivalence_test_tost_concludes_equivalence_for_a_tight_null_effect():
    differences = np.array([0.01, -0.02, 0.015, -0.01, 0.005, -0.005])
    result = equivalence_test_tost(differences, margin=1.0)
    assert result["equivalent"]


def test_equivalence_test_tost_rejects_equivalence_for_a_large_effect():
    differences = np.array([5.0, 5.2, 4.8, 5.1, 4.9, 5.0])
    result = equivalence_test_tost(differences, margin=1.0)
    assert not result["equivalent"]


def test_equivalence_test_tost_rejects_non_positive_margin():
    with pytest.raises(ValueError, match="margin must be positive"):
        equivalence_test_tost(np.array([0.0, 0.1]), margin=0.0)


def test_power_curve_by_resampling_is_reproducible_and_bounded():
    subject_ids = [f"s{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.0, scale=0.5, size=20)  # clearly nonzero effect

    def reject_null(sample: np.ndarray) -> bool:
        return len(sample) >= 2 and stats_ttest_rejects(sample)

    def stats_ttest_rejects(sample: np.ndarray) -> bool:
        from scipy import stats as scipy_stats
        _, p_value = scipy_stats.ttest_1samp(sample, popmean=0.0)
        return bool(p_value < 0.05)

    first = power_curve_by_resampling(
        values, subject_ids, sample_sizes=[5, 10, 20], reject_null_fn=reject_null, n_resamples=200, seed=1,
    )
    second = power_curve_by_resampling(
        values, subject_ids, sample_sizes=[5, 10, 20], reject_null_fn=reject_null, n_resamples=200, seed=1,
    )
    pd.testing.assert_frame_equal(first, second)
    assert (first["power"] >= 0).all() and (first["power"] <= 1).all()


# --- gates.py --------------------------------------------------------------------------

def test_check_calibration_gate_passes_within_thresholds():
    result = check_calibration_gate(0.03, 0.08)
    assert result.passed
    assert result.reasons["false_positive_rate_ok"]
    assert result.reasons["upper_ci_bound_ok"]


def test_check_calibration_gate_fails_when_fpr_too_high():
    result = check_calibration_gate(0.10, 0.08)
    assert not result.passed
    assert not result.reasons["false_positive_rate_ok"]


def test_check_error_difference_favors_method():
    assert check_error_difference_favors_method(0.5)
    assert not check_error_difference_favors_method(0.0)
    assert not check_error_difference_favors_method(-0.5)


def test_check_bootstrap_excludes_zero():
    assert check_bootstrap_excludes_zero({"excludes_zero": True})
    assert not check_bootstrap_excludes_zero({"excludes_zero": False})


def test_check_practically_meaningful_improvement_boundary():
    assert check_practically_meaningful_improvement(20.0)
    assert check_practically_meaningful_improvement(25.0)
    assert not check_practically_meaningful_improvement(19.9)


def test_check_false_positive_rate_no_worse():
    assert check_false_positive_rate_no_worse(0.05, 0.05)
    assert check_false_positive_rate_no_worse(0.04, 0.05)
    assert not check_false_positive_rate_no_worse(0.06, 0.05)


def test_check_consistent_direction_across_cohorts_agrees():
    assert check_consistent_direction_across_cohorts({"nki": 0.5, "ppmi": 0.3})
    assert not check_consistent_direction_across_cohorts({"nki": 0.5, "ppmi": -0.3})


def test_check_consistent_direction_across_cohorts_requires_two_cohorts():
    with pytest.raises(ValueError, match="at least two cohorts"):
        check_consistent_direction_across_cohorts({"nki": 0.5})


def test_check_not_single_seed_driven_passes_for_balanced_seeds():
    assert check_not_single_seed_driven(np.array([1.0, 1.1, 0.9, 1.05, 0.95]))


def test_check_not_single_seed_driven_fails_when_one_seed_dominates_share():
    # seed 0 alone accounts for >50% of the total effect magnitude
    assert not check_not_single_seed_driven(np.array([10.0, 0.5, 0.5, 0.5]))


def test_check_not_single_seed_driven_fails_on_sign_flip_even_under_share_limit():
    # dominant seed's share of total magnitude is ~24% (<=0.5), but removing it
    # flips the mean's sign from negative to positive.
    per_seed_effect = np.array([-2.30, -4.59, -4.83, 3.13, 4.13, 1.07])
    assert not check_not_single_seed_driven(per_seed_effect)


def test_check_not_single_seed_driven_rejects_single_seed():
    with pytest.raises(ValueError, match="at least two seeds"):
        check_not_single_seed_driven(np.array([1.0]))


def test_evaluate_superiority_claim_true_when_all_six_criteria_hold():
    result = evaluate_superiority_claim(
        paired_error_difference=0.5,
        bootstrap_ci={"excludes_zero": True},
        relative_improvement_pct=25.0,
        method_fpr=0.03,
        comparator_fpr=0.05,
        direction_by_cohort={"nki": 0.4, "ppmi": 0.6},
        per_seed_effect=np.array([1.0, 1.1, 0.9, 1.05, 0.95]),
    )
    assert result.superior
    assert all(result.checks.values())


def test_evaluate_superiority_claim_false_when_one_criterion_fails():
    result = evaluate_superiority_claim(
        paired_error_difference=0.5,
        bootstrap_ci={"excludes_zero": True},
        relative_improvement_pct=10.0,  # below the 20% threshold
        method_fpr=0.03,
        comparator_fpr=0.05,
        direction_by_cohort={"nki": 0.4, "ppmi": 0.6},
        per_seed_effect=np.array([1.0, 1.1, 0.9, 1.05, 0.95]),
    )
    assert not result.superior
    assert not result.checks["practically_meaningful"]
    assert result.checks["error_difference_favors_method"]  # other criteria unaffected


def test_check_invariant_variant_justified_passes_when_retained_and_improved():
    result = check_invariant_variant_justified(
        invariant_dose_response_auroc=0.80,
        raw_dose_response_auroc=0.83,  # 0.03 drop, within the 0.05 budget
        invariant_calibration_margin=0.06,
        raw_calibration_margin=0.02,
    )
    assert result.passed


def test_check_invariant_variant_justified_fails_when_dose_response_drops_too_much():
    result = check_invariant_variant_justified(
        invariant_dose_response_auroc=0.70,
        raw_dose_response_auroc=0.83,  # 0.13 drop, exceeds the 0.05 budget
        invariant_calibration_margin=0.06,
        raw_calibration_margin=0.02,
    )
    assert not result.passed
    assert not result.reasons["dose_response_retained"]


def test_check_invariant_variant_justified_fails_when_margin_not_improved():
    result = check_invariant_variant_justified(
        invariant_dose_response_auroc=0.82,
        raw_dose_response_auroc=0.83,
        invariant_calibration_margin=0.02,
        raw_calibration_margin=0.02,  # no improvement
    )
    assert not result.passed
    assert not result.reasons["calibration_margin_improved"]


# --- config.py -------------------------------------------------------------------------

def test_geometric_config_loads_and_passes_its_own_lock():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    assert config["study"] == "geometric_longitudinal_validation"
    assert config["cnn"]["seeds"] == [0, 1, 2, 3, 4]


def test_geometric_config_rejects_architecture_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["cnn"]["architectures"] = ["double_conv"]
    with pytest.raises(ValueError, match="architectures"):
        validate_locked_config(changed)


def test_geometric_config_rejects_variant_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["cnn"]["variants"] = ["raw"]
    with pytest.raises(ValueError, match="variants"):
        validate_locked_config(changed)


def test_geometric_config_rejects_missing_target():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    del changed["deformation"]["targets"]["cerebellar_white_matter"]
    with pytest.raises(ValueError, match="targets"):
        validate_locked_config(changed)


def test_geometric_config_rejects_k_null_resamples_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["invariance"]["self_calibration"]["k_null_resamples"] = 10
    with pytest.raises(ValueError, match="k_null_resamples"):
        validate_locked_config(changed)


def test_geometric_config_rejects_calibration_gate_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["decision_gates"]["calibration"]["fpr_max"] = 0.20
    with pytest.raises(ValueError, match="fpr_max"):
        validate_locked_config(changed)


def test_geometric_config_rejects_superiority_gate_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["decision_gates"]["superiority"]["min_improvement_pct"] = 5.0
    with pytest.raises(ValueError, match="min_improvement_pct"):
        validate_locked_config(changed)


def test_geometric_config_rejects_invariant_vs_raw_gate_drift():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["decision_gates"]["invariant_vs_raw"]["max_auroc_reduction"] = 0.50
    with pytest.raises(ValueError, match="max_auroc_reduction"):
        validate_locked_config(changed)


def test_geometric_config_reports_multiple_errors_together():
    config = load_geometric_config(ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    changed = json.loads(json.dumps(config))
    changed["cnn"]["architectures"] = ["double_conv"]
    changed["cnn"]["seeds"] = [0]
    with pytest.raises(ValueError) as excinfo:
        validate_locked_config(changed)
    assert "architectures" in str(excinfo.value)
    assert "seeds" in str(excinfo.value)


# --- frozen_cnn.py ----------------------------------------------------------------------
# 32^3 volumes, as in the trained_cnn.py section, to survive the backbone's pooling.

FROZEN_SHAPE = (32, 32, 32)
FROZEN_CENTER = (16, 16, 16)


def test_build_frozen_cnn_returns_frozen_eval_model():
    model = build_frozen_cnn("double_conv", ["t1"], seed=0, device="cpu")
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.out_features == 7680


def test_extract_features_shape_matches_model_out_features():
    model = build_frozen_cnn("double_conv", ["t1"], seed=0, device="cpu")
    channels = np.random.default_rng(0).uniform(0, 1, size=(1, *FROZEN_SHAPE)).astype(np.float32)
    features = extract_features(model, channels, device="cpu")
    assert features.shape == (model.out_features,)


def test_feature_change_magnitude_matches_manual_l2_norm():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 6.0, 3.0], dtype=np.float32)
    expected = float(np.linalg.norm(b - a))
    assert feature_change_magnitude(a, b) == pytest.approx(expected)
    assert feature_change_magnitude(a, a) == pytest.approx(0.0)


def test_feature_change_magnitude_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        feature_change_magnitude(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_build_response_fn_gives_zero_for_an_identical_pair():
    mask = _sphere_mask(FROZEN_SHAPE, FROZEN_CENTER, radius=10)
    baseline = (mask.astype(np.float32) * 80.0)
    model = build_frozen_cnn("double_conv", ["t1"], seed=0, device="cpu")
    response_fn = build_response_fn(
        model, mask, variant="raw", sigma_voxels=2.0, scaler="minmax", channels=["t1"],
    )
    assert response_fn(baseline, baseline.copy()) == pytest.approx(0.0, abs=1e-5)


def test_build_response_fn_raw_and_invariant_differ_for_a_real_deformation():
    mask = _sphere_mask(FROZEN_SHAPE, FROZEN_CENTER, radius=10)
    rng = np.random.default_rng(1)
    baseline = (mask.astype(np.float32) * 80.0 + rng.uniform(0, 5, size=FROZEN_SHAPE).astype(np.float32))
    field = radial_bump_field(FROZEN_SHAPE, mask, magnitude=0.3, sigma_voxels=4.0, direction="contract")
    followup, _ = synthesize_followup(baseline, mask, field)

    model = build_frozen_cnn("double_conv", ["t1"], seed=0, device="cpu")
    raw_response_fn = build_response_fn(
        model, mask, variant="raw", sigma_voxels=2.0, scaler="minmax", channels=["t1"],
    )
    invariant_response_fn = build_response_fn(
        model, mask, variant="invariant", sigma_voxels=2.0, scaler="minmax", channels=["t1"],
    )
    raw_response = raw_response_fn(baseline, followup)
    invariant_response = invariant_response_fn(baseline, followup)
    assert raw_response > 0.0
    assert invariant_response > 0.0
    assert raw_response != pytest.approx(invariant_response)


def test_build_response_fn_is_deterministic():
    mask = _sphere_mask(FROZEN_SHAPE, FROZEN_CENTER, radius=10)
    rng = np.random.default_rng(2)
    baseline = (mask.astype(np.float32) * 80.0 + rng.uniform(0, 5, size=FROZEN_SHAPE).astype(np.float32))
    field = radial_bump_field(FROZEN_SHAPE, mask, magnitude=0.2, sigma_voxels=4.0, direction="contract")
    followup, _ = synthesize_followup(baseline, mask, field)
    model = build_frozen_cnn("double_conv", ["t1"], seed=0, device="cpu")
    response_fn = build_response_fn(
        model, mask, variant="invariant", sigma_voxels=2.0, scaler="minmax", channels=["t1"],
    )
    assert response_fn(baseline, followup) == pytest.approx(response_fn(baseline, followup))


# --- decoder.py -------------------------------------------------------------------------

def _linear_calibration_data(n: int, *, seed: int, true_coef: np.ndarray, noise: float = 0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(true_coef)))
    y = X @ true_coef + rng.normal(scale=noise, size=n)
    subject_ids = [f"sub-{i}" for i in range(n)]
    return X, y, subject_ids


TRUE_COEF = np.array([2.0, -1.0, 0.5])


def test_select_ridge_alpha_rejects_single_subject():
    X, y, subject_ids = _linear_calibration_data(1, seed=0, true_coef=TRUE_COEF)
    with pytest.raises(ValueError, match="at least two calibration subjects"):
        select_ridge_alpha(X, y, np.asarray(subject_ids), alphas=[0.1, 1.0])


def test_fit_decoder_rejects_mismatched_lengths():
    X, y, subject_ids = _linear_calibration_data(10, seed=0, true_coef=TRUE_COEF)
    with pytest.raises(ValueError, match="same length"):
        fit_decoder(X, y[:-1], subject_ids, alphas=[0.1, 1.0])


def test_fit_decoder_recovers_a_clean_linear_relationship():
    X_cal, y_cal, subjects_cal = _linear_calibration_data(24, seed=0, true_coef=TRUE_COEF, noise=0.1)
    X_val, y_val, _ = _linear_calibration_data(15, seed=1, true_coef=TRUE_COEF, noise=0.1)

    result = fit_decoder(X_cal, y_cal, subjects_cal, alphas=[0.01, 0.1, 1.0, 10.0], n_folds=4)
    predictions = decoder_predict(result, X_val)
    metrics = regression_accuracy_metrics(y_val, predictions)

    assert result.n_calibration_subjects == 24
    assert metrics["mae"] < 0.5
    assert metrics["calibration_slope"] == pytest.approx(1.0, abs=0.2)


def test_bootstrap_prediction_intervals_are_well_ordered_and_reproducible():
    X_cal, y_cal, subjects_cal = _linear_calibration_data(24, seed=0, true_coef=TRUE_COEF, noise=0.1)
    X_val, y_val, _ = _linear_calibration_data(10, seed=2, true_coef=TRUE_COEF, noise=0.1)
    result = fit_decoder(X_cal, y_cal, subjects_cal, alphas=[0.01, 0.1, 1.0, 10.0], n_folds=4)

    lower1, upper1 = bootstrap_prediction_intervals(
        X_cal, y_cal, subjects_cal, X_val, alpha=result.alpha, n_bootstrap=200, seed=5,
    )
    lower2, upper2 = bootstrap_prediction_intervals(
        X_cal, y_cal, subjects_cal, X_val, alpha=result.alpha, n_bootstrap=200, seed=5,
    )
    np.testing.assert_array_equal(lower1, lower2)
    np.testing.assert_array_equal(upper1, upper2)
    assert (lower1 <= upper1).all()
    # a well-calibrated interval should cover most true values even with few subjects
    assert interval_coverage(y_val, lower1, upper1) > 0.3


def test_bootstrap_prediction_intervals_rejects_single_subject():
    X_cal, y_cal, _ = _linear_calibration_data(5, seed=0, true_coef=TRUE_COEF)
    with pytest.raises(ValueError, match="at least two calibration subjects"):
        bootstrap_prediction_intervals(
            X_cal, y_cal, ["only-one"] * 5, X_cal[:2], alpha=1.0, n_bootstrap=10, seed=1,
        )


# --- run_geometric_validation.py --------------------------------------------------------

GEOMETRIC_CONFIG_PATH = ROOT / "configs" / "geometric_longitudinal_validation.yaml"


def test_deterministic_pilot_validation_split_is_reproducible_and_disjoint():
    ids = [f"sub-{index:03d}" for index in range(30)]
    first = deterministic_pilot_validation_split(ids, n_validation=10, n_pilot=8, seed=3)
    second = deterministic_pilot_validation_split(list(reversed(ids)), n_validation=10, n_pilot=8, seed=3)
    pd.testing.assert_frame_equal(first, second)
    pilot = set(first[first["split"] == "pilot"]["subject_id"])
    validation = set(first[first["split"] == "validation"]["subject_id"])
    assert len(pilot) == 8
    assert len(validation) == 10
    assert pilot.isdisjoint(validation)


def test_deterministic_pilot_validation_split_rejects_too_few_subjects():
    with pytest.raises(ValueError, match="only"):
        deterministic_pilot_validation_split(["a", "b", "c"], n_validation=5, n_pilot=5, seed=1)


def test_planned_cells_cardinality_matches_the_preregistered_grid():
    config = load_geometric_config(GEOMETRIC_CONFIG_PATH)
    cells = _planned_cells(config)
    calibration_cells = [cell for cell in cells if cell["kind"] == "calibrate_invariance"]
    train_cells = [cell for cell in cells if cell["kind"] == "train"]
    cnn_cells = [cell for cell in cells if cell["kind"] == "cnn"]
    # 2 cohorts x 2 architectures x 5 seeds
    assert len(calibration_cells) == 2 * 2 * 5
    # 2 cohorts x 2 architectures x 2 variants x 5 seeds
    assert len(train_cells) == 2 * 2 * 2 * 5
    # 2 cohorts x 2 architectures x 2 targets x 5 seeds x 2 variants x 2 model kinds
    assert len(cnn_cells) == 2 * 2 * 2 * 5 * 2 * 2
    assert len(cells) == len(calibration_cells) + len(train_cells) + len(cnn_cells)


def test_target_relevant_conditions_selects_sham_plus_matching_target():
    config = load_geometric_config(GEOMETRIC_CONFIG_PATH)
    hippocampus_conditions = target_relevant_conditions(config, "hippocampus")
    cerebellar_conditions = target_relevant_conditions(config, "cerebellar_white_matter")
    assert {"exact_duplicate", "resampling_sham"}.issubset(hippocampus_conditions)
    assert {"exact_duplicate", "resampling_sham"}.issubset(cerebellar_conditions)
    assert len(hippocampus_conditions) == 2 + 4  # contract 2/5/10%, expand 5%
    assert len(cerebellar_conditions) == 2 + 1  # contract 5% only


def test_channels_for_variant_matches_configured_channel_count():
    config = load_geometric_config(GEOMETRIC_CONFIG_PATH)
    mask = _sphere_mask((16, 16, 16), (8, 8, 8), radius=6)
    image = (mask.astype(np.float32) * 80.0)
    channels_out = _channels_for_variant(image, mask, "raw", 2.0, config["preprocessing"])
    assert channels_out.shape[0] == len(config["preprocessing"]["channels"])


def test_channels_for_variant_raw_and_invariant_differ():
    config = load_geometric_config(GEOMETRIC_CONFIG_PATH)
    mask = _sphere_mask((16, 16, 16), (8, 8, 8), radius=6)
    rng = np.random.default_rng(0)
    image = (mask.astype(np.float32) * 80.0 + rng.uniform(0, 10, size=(16, 16, 16)).astype(np.float32))
    raw_channels = _channels_for_variant(image, mask, "raw", 2.0, config["preprocessing"])
    invariant_channels = _channels_for_variant(image, mask, "invariant", 2.0, config["preprocessing"])
    assert not np.array_equal(raw_channels, invariant_channels)


def test_process_condition_exact_duplicate_has_zero_ground_truth():
    shape = (21, 21, 21)
    mask = _sphere_mask(shape, (10, 10, 10), radius=6)
    baseline = mask.astype(np.float32) * 80.0
    result = process_condition(
        baseline, mask,
        condition_name="exact_duplicate", condition_spec={}, region_mask=None,
        response_fn=lambda a, b: float(np.abs(a - b).sum()),
        seed=0, magnitude_bounds=(0.0, 1.0),
        qc_config={"tolerance_pct": 0.5, "max_outside_change_pct": 2.0, "max_boundary_gradient": 1.0},
        target_sigma_voxels=4.0,
    )
    assert result["requested_change_pct"] == 0.0
    assert result["realized_change_pct"] == pytest.approx(0.0, abs=1e-4)
    assert result["qc_passed"]
    assert result["response"] == pytest.approx(0.0, abs=1e-4)


def test_process_condition_resampling_sham_has_zero_ground_truth_but_nonzero_response():
    shape = (21, 21, 21)
    mask = _sphere_mask(shape, (10, 10, 10), radius=6)
    rng = np.random.default_rng(0)
    baseline = (mask.astype(np.float32) * 80.0 + rng.uniform(0, 5, size=shape).astype(np.float32))
    result = process_condition(
        baseline, mask,
        condition_name="resampling_sham",
        condition_spec={"max_translation_voxels": 0.5, "max_rotation_degrees": 1.0},
        region_mask=None,
        response_fn=lambda a, b: float(np.abs(a - b).sum()),
        seed=1, magnitude_bounds=(0.0, 1.0),
        qc_config={"tolerance_pct": 0.5, "max_outside_change_pct": 5.0, "max_boundary_gradient": 5.0},
        target_sigma_voxels=4.0,
    )
    assert result["requested_change_pct"] == 0.0
    assert result["realized_change_pct"] == pytest.approx(0.0, abs=1e-2)


def test_process_condition_hippocampus_contraction_matches_requested_amplitude():
    shape = (21, 21, 21)
    region_mask = _sphere_mask(shape, (10, 10, 10), radius=5)
    baseline = region_mask.astype(np.float32) * 80.0
    result = process_condition(
        baseline, region_mask,
        condition_name="hippocampus_contract_05",
        condition_spec={"target": "hippocampus", "direction": "contract", "target_change_pct": -5.0},
        region_mask=region_mask,
        response_fn=lambda a, b: float(np.abs(a - b).sum()),
        seed=0, magnitude_bounds=(0.0, 0.5),
        qc_config={"tolerance_pct": 0.5, "max_outside_change_pct": 5.0, "max_boundary_gradient": 5.0},
        target_sigma_voxels=4.0,
    )
    assert result["requested_change_pct"] == -5.0
    assert result["realized_change_pct"] == pytest.approx(-5.0, abs=0.5)
    assert result["response"] > 0.0


def test_prepare_runs_against_real_cohort_data(tmp_path):
    config = yaml.safe_load(GEOMETRIC_CONFIG_PATH.read_text())
    config["output"]["root"] = str(tmp_path / "geometric_validation")
    temp_config_path = tmp_path / "config.yaml"
    temp_config_path.write_text(yaml.safe_dump(config))

    run_root = prepare(temp_config_path, "test_run")
    assert run_root.is_dir()
    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    assert set(assignment["cohort"]) <= {"nki", "ppmi"}
    for cohort, group in assignment.groupby("cohort"):
        assert (group["split"] == "pilot").sum() == config["subjects"]["pilot_per_cohort"], cohort
        assert (group["split"] == "validation").sum() == config["subjects"]["validation_per_cohort"], cohort
        assert group["subject_id"].is_unique

    manifest = json.loads((run_root / "completion_manifest.json").read_text())
    assert manifest["status"] == "prepared"
    assert len(manifest["planned_cells"]) == len(_planned_cells(load_geometric_config(temp_config_path)))
    assert (run_root / "environment.json").is_file()
    assert (run_root / "resolved_config.yaml").is_file()


def test_prepare_refuses_to_overwrite_an_existing_run(tmp_path):
    config = yaml.safe_load(GEOMETRIC_CONFIG_PATH.read_text())
    config["output"]["root"] = str(tmp_path / "geometric_validation")
    temp_config_path = tmp_path / "config.yaml"
    temp_config_path.write_text(yaml.safe_dump(config))

    prepare(temp_config_path, "dup_run")
    with pytest.raises(FileExistsError):
        prepare(temp_config_path, "dup_run")


def test_resolve_sigma_voxels_raw_variant_never_needs_calibration(tmp_path):
    # No calibration cell exists under tmp_path at all -- raw must not require one.
    assert resolve_sigma_voxels(
        tmp_path, cohort_name="nki", architecture="double_conv", seed=0, variant="raw",
    ) == 0.0


def test_resolve_sigma_voxels_invariant_variant_requires_calibration_cell(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate-invariance-cell"):
        resolve_sigma_voxels(
            tmp_path, cohort_name="nki", architecture="double_conv", seed=0, variant="invariant",
        )


def test_resolve_sigma_voxels_invariant_variant_reads_the_calibrated_value(tmp_path):
    cell_dir = tmp_path / "invariance_calibration" / "nki" / "double_conv" / "seed_0"
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell_manifest.json").write_text(json.dumps({"chosen_sigma_voxels": 2.0}))
    sigma = resolve_sigma_voxels(
        tmp_path, cohort_name="nki", architecture="double_conv", seed=0, variant="invariant",
    )
    assert sigma == pytest.approx(2.0)
