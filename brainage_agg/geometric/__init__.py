"""Geometric longitudinal validation: measurement-invariance-engineered confirmatory study.

See geometric_longitudinal_validation_architecture.md for the design this package
implements and geometric_longitudinal_validation_plan.md for the scientific protocol.
"""

from .config import LOCKED as CONFIG_LOCKED
from .config import load_geometric_config, validate_locked_config
from .decoder import (
    DecoderFitResult,
    bootstrap_prediction_intervals,
    fit_decoder,
    select_ridge_alpha,
)
from .decoder import predict as decoder_predict
from .frozen_cnn import (
    build_frozen_cnn,
    build_response_fn,
    extract_features,
    feature_change_magnitude,
)
from .gates import (
    GateResult,
    SuperiorityClaimResult,
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
from .deformation import (
    DeformationField,
    calibrate_magnitude_for_target_change,
    global_scaling_field,
    identity_field,
    jacobian_determinant,
    radial_bump_field,
    realized_volume_change_pct,
    rigid_subvoxel_field,
)
from .determinism import (
    assert_float32,
    enforce_deterministic_environment,
    preprocess_pair,
    reproducibility_self_test,
)
from .invariance_layer import (
    FilterCalibrationResult,
    apply_invariance_layer,
    band_limit_filter,
    calibrate_filter_cutoff,
    preprocess_pair_with_variant,
)
from .qc import DeformationQCResult, evaluate_deformation_qc
from .self_calibration import (
    SelfCalibrationResult,
    compute_null_response_distribution,
    compute_self_calibration,
    null_resample_fields,
    zscore_against_null,
)
from .self_calibration import to_dataframe as self_calibration_to_dataframe
from .statistics import (
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
from .synthesis import invert_displacement_field, synthesize_followup, warp_image
from .trained_cnn import (
    TrainingExample,
    TrainingResult,
    assert_no_training_subject_overlap,
    load_trained_checkpoint,
    save_checkpoint,
    train_cnn,
)

__all__ = [
    "CONFIG_LOCKED",
    "DecoderFitResult",
    "DeformationField",
    "DeformationQCResult",
    "FilterCalibrationResult",
    "apply_invariance_layer",
    "assert_float32",
    "band_limit_filter",
    "bootstrap_prediction_intervals",
    "build_frozen_cnn",
    "build_response_fn",
    "calibrate_filter_cutoff",
    "calibrate_magnitude_for_target_change",
    "decoder_predict",
    "enforce_deterministic_environment",
    "evaluate_deformation_qc",
    "extract_features",
    "feature_change_magnitude",
    "global_scaling_field",
    "identity_field",
    "invert_displacement_field",
    "jacobian_determinant",
    "load_geometric_config",
    "preprocess_pair",
    "preprocess_pair_with_variant",
    "radial_bump_field",
    "realized_volume_change_pct",
    "validate_locked_config",
    "GateResult",
    "SelfCalibrationResult",
    "SuperiorityClaimResult",
    "TrainingExample",
    "TrainingResult",
    "assert_no_training_subject_overlap",
    "calibration_slope_intercept",
    "check_bootstrap_excludes_zero",
    "check_calibration_gate",
    "check_consistent_direction_across_cohorts",
    "check_error_difference_favors_method",
    "check_false_positive_rate_no_worse",
    "check_invariant_variant_justified",
    "check_not_single_seed_driven",
    "check_practically_meaningful_improvement",
    "compute_null_response_distribution",
    "compute_self_calibration",
    "equivalence_test_tost",
    "evaluate_superiority_claim",
    "interval_coverage",
    "load_trained_checkpoint",
    "null_resample_fields",
    "paired_change_test",
    "power_curve_by_resampling",
    "regression_accuracy_metrics",
    "reproducibility_self_test",
    "rigid_subvoxel_field",
    "save_checkpoint",
    "select_ridge_alpha",
    "self_calibration_to_dataframe",
    "sign_accuracy",
    "spearman_dose_response",
    "subject_clustered_bootstrap_ci",
    "subject_level_discrimination_auroc",
    "synthesize_followup",
    "train_cnn",
    "warp_image",
    "zscore_against_null",
]
