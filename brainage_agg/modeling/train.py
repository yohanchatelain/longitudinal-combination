"""Per-cell evaluation: one (extractor × arm × aggregation × band) combination.

Returns a list of per-fold result dicts, ready to be appended to results.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from brainage_agg.agg.aggregations import AGGREGATION_SPECS, validate_arm
from brainage_agg.features.loader import load_features, align_to_manifest, get_extractor_id, is_fs_roi_file
from brainage_agg.modeling.cv import run_nested_cv


def evaluate_cell(
    npz_path: str | Path,
    manifest: pd.DataFrame,
    arm: str,
    aggregation: str,
    band: str,
    cv_config: dict,
) -> list[dict]:
    """Run nested CV for one factorial cell and return per-fold rows.

    Args:
        npz_path:   path to the feature .npz file
        manifest:   subject manifest from build_manifest
        arm:        'A' or 'B'
        aggregation: one of AGGREGATION_SPECS keys
        band:       'all', 'Child', or 'Adult'
        cv_config:  dict with keys n_outer_folds, n_inner_folds, ridge_alphas,
                    lme_threshold

    Returns:
        list of dicts with schema:
          target_arm, extractor, cnn_arch, scaler, channels, cnn_seed,
          aggregation, age_band, fold, MAE, RMSE, R2, r, chance_mae,
          n_train, n_test, n_subjects_train, n_subjects_test, best_alpha
    """
    # Validate before loading (fast fail)
    validate_arm(aggregation, arm)

    # Load and align features
    X, meta_df, file_meta = load_features(npz_path)
    subject_data = align_to_manifest(X, meta_df, manifest)

    # Run nested CV
    fold_rows = run_nested_cv(
        manifest=manifest,
        subject_data=subject_data,
        aggregation=aggregation,
        arm=arm,
        band=band,
        n_outer_folds=cv_config["n_outer_folds"],
        n_inner_folds=cv_config["n_inner_folds"],
        ridge_alphas=cv_config["ridge_alphas"],
        lme_threshold=cv_config.get("lme_threshold", 700),
    )

    # Annotate rows with extractor metadata
    extractor_type = "fs_roi" if is_fs_roi_file(npz_path) else "cnn"
    extractor_id = get_extractor_id(npz_path)

    # eTIV is a FreeSurfer-derived measure correlated with FS ROI features.
    # Flag any FS ROI cell in Arm B2 so downstream analysis can caveat or exclude it.
    leakage_warning = extractor_type == "fs_roi" and arm.upper() in ("B", "B2")

    result_rows = []
    for fr in fold_rows:
        row = {
            "target_arm": arm,
            "leakage_warning": leakage_warning,
            "extractor_type": extractor_type,
            "extractor": extractor_id,
            "cnn_arch": file_meta.get("cnn_arch", ""),
            "scaler": file_meta.get("scaler", ""),
            "channels": file_meta.get("channels", ""),
            "cnn_seed": file_meta.get("seed", ""),
            "aggregation": aggregation,
            "age_band": band,
            "fold": fr["fold"],
            "MAE": fr["MAE"],
            "RMSE": fr["RMSE"],
            "R2": fr["R2"],
            "r": fr["r"],
            "chance_mae": fr["chance_mae"],
            "best_alpha": fr["best_alpha"],
            "n_train": fr["n_train"],
            "n_test": fr["n_test"],
            "n_subjects_train": fr["n_subjects_train"],
            "n_subjects_test": fr["n_subjects_test"],
            "per_subject": json.dumps(fr.get("per_subject", [])),
        }
        result_rows.append(row)

    return result_rows
