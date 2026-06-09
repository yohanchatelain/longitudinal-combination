"""Nested CV with age-band-stratified outer folds.

Outer folds estimate generalisation performance; StratifiedGroupKFold keeps
Child/Adult proportions matched across folds while preserving subject grouping.
Inner folds select Ridge alpha within each outer training set.

Leakage rules enforced here:
- StandardScaler fit on train subjects only.
- LME estimator fit on train subjects only.
- Ridge alpha selection uses inner CV on train subjects only.
- Subject disjointness asserted at every outer fold.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from brainage_agg.agg import aggregations as agg_mod
from brainage_agg.agg.lme import make_lme_estimator


def _build_agg_matrix(
    subject_ids: np.ndarray,
    subject_data: dict[str, dict],
    manifest: pd.DataFrame,
    aggregation: str,
    arm: str,
    lme_subjects: np.ndarray | None = None,
    lme_slopes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aggregated feature matrix X_agg (n_subjects × d) and target y.

    Also returns the target vector:
      Arm A: mean age across subject's timepoints
      Arm B: brain_change_rate = PC1 of annualized FS ROI feature change (from manifest)
    """
    rows_X = []
    rows_y = []
    sid_manifest = manifest.set_index("subject_id")

    for sid in subject_ids:
        data = subject_data[sid]
        feats = data["features"]  # (n_tp, n_feat) — already z-scored
        ages = data["ages"]
        mrow = sid_manifest.loc[sid]

        # Build target
        if arm.upper() == "A":
            y_val = float(np.mean(ages))
        elif arm.upper() == "B":
            bcr = mrow["brain_change_rate"]
            if pd.isna(bcr):
                raise ValueError(
                    f"brain_change_rate missing for subject {sid}. "
                    "Call compute_brain_change_rate() and merge the result into the manifest "
                    "before running Arm B."
                )
            y_val = float(bcr)
        else:  # Arm B2: annualized eTIV rate
            etiv_r = mrow["etiv_rate"]
            if pd.isna(etiv_r):
                raise ValueError(
                    f"etiv_rate missing for subject {sid}. "
                    "Call compute_etiv_rate() and merge the result into the manifest "
                    "before running Arm B2."
                )
            y_val = float(etiv_r)

        # Build aggregation
        if aggregation == "mean":
            x = agg_mod.mean(feats)
        elif aggregation == "concatenation":
            t_first_local = 0
            t_last_local = len(feats) - 1
            t_mid_global = int(mrow["t_mid_idx"])
            if t_mid_global < 0:
                # 2-timepoint subjects: concatenate first and last only
                x = np.concatenate([feats[t_first_local], feats[t_last_local]])
            else:
                global_row_indices = np.asarray(mrow["row_indices"] if isinstance(mrow["row_indices"], list) else list(mrow["row_indices"]))
                t_mid_local_arr = np.where(global_row_indices == t_mid_global)[0]
                if len(t_mid_local_arr) == 0:
                    raise ValueError(f"t_mid_idx {t_mid_global} not found in row_indices for subject {sid}")
                t_mid_local = int(t_mid_local_arr[0])
                x = agg_mod.concatenation(feats, t_first_local, t_mid_local, t_last_local)
        elif aggregation == "annualized_rate":
            x = agg_mod.annualized_rate(feats, ages)
        elif aggregation in ("lme_slope", "lme_slope_change"):
            if lme_subjects is None or lme_slopes is None:
                raise RuntimeError("LME transform output required but not provided.")
            x = agg_mod.lme_slope(sid, lme_subjects, lme_slopes)
        elif aggregation == "difference":
            x = agg_mod.difference(feats)
        else:
            raise ValueError(f"Unknown aggregation: '{aggregation}'")

        rows_X.append(x)
        rows_y.append(y_val)

    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def run_nested_cv(
    manifest: pd.DataFrame,
    subject_data: dict[str, dict],
    aggregation: str,
    arm: str,
    band: str,
    n_outer_folds: int,
    n_inner_folds: int,
    ridge_alphas: list[float],
    lme_threshold: int = 700,
) -> list[dict]:
    """Run nested GroupKFold CV and return per-fold metric dicts.

    Args:
        manifest:       subject manifest (from build_manifest)
        subject_data:   {subject_id: {features, ages, row_indices}} aligned to manifest
        aggregation:    name of aggregation to apply
        arm:            'A', 'B', or 'B2'
        band:           'all', 'Child', or 'Adult' — which subjects to include
        n_outer_folds:  number of outer CV folds
        n_inner_folds:  number of inner CV folds for alpha selection
        ridge_alphas:   list of alpha values for Ridge grid search
        lme_threshold:  feature count threshold for StatsmodelsLME vs VectorizedOLS

    Returns:
        list of dicts, one per outer fold, with keys:
          fold, MAE, RMSE, R2, r, n_train, n_test, n_subjects_train, n_subjects_test,
          best_alpha, aggregation, arm, band, chance_mae
    """
    # --- Filter subjects by cohort membership and band ---
    agg_mod.validate_arm(aggregation, arm)
    spec = agg_mod.AGGREGATION_SPECS[aggregation]

    df = manifest.copy()
    # cohort_span1 supersedes cohort_all when present (follow-up span filter applied at manifest build)
    cohort_col = "cohort_span1" if "cohort_span1" in df.columns else "cohort_all"
    # Cohort filter — fall back to cohort_all when cohort_concat is empty
    # (e.g. PPMI where all subjects have exactly 2 sessions → 2-point concat)
    if spec["cohort"] == "concat":
        df_concat = df[df["cohort_concat"]]
        df = df_concat if not df_concat.empty else df[df[cohort_col]]
    else:
        df = df[df[cohort_col]]
    # Arm B, B2 and annualized_rate: exclude delta_t == 0
    # (annualized_rate divides by delta_t; it must exclude zero-interval subjects)
    if arm.upper() in ("B", "B2") or aggregation == "annualized_rate":
        df = df[~df["exclude_arm_b"]]
    # Band filter
    if band != "all":
        df = df[df["band"] == band]

    if len(df) < n_outer_folds * 2:
        return []

    subject_ids = df["subject_id"].to_numpy()

    # Build stratification labels for the outer splitter.
    # For "all" band: use the band column so Child/Adult proportions are
    # preserved across folds. For single-band slices every label is identical,
    # so StratifiedGroupKFold degenerates to GroupKFold — no harm done.
    if "band" in df.columns:
        strat_labels = df["band"].fillna("Unknown").to_numpy()
    else:
        strat_labels = np.zeros(len(subject_ids), dtype=int)

    # Detect feature dimensionality from first subject
    first_sid = subject_ids[0]
    n_features = subject_data[first_sid]["features"].shape[1]

    dummy_X = np.zeros((len(subject_ids), 1))
    sgkf_outer = StratifiedGroupKFold(n_splits=n_outer_folds)

    fold_results = []

    for fold_idx, (train_local, test_local) in enumerate(
        sgkf_outer.split(dummy_X, y=strat_labels, groups=subject_ids)
    ):
        train_sids = subject_ids[train_local]
        test_sids = subject_ids[test_local]

        # --- Disjointness assertion (non-negotiable) ---
        assert set(train_sids).isdisjoint(set(test_sids)), (
            f"LEAKAGE: train/test subjects overlap in fold {fold_idx}"
        )

        # --- Step 1: z-score features from train subjects only ---
        # Build flat train matrix for scaler fitting
        train_X_flat = np.vstack([subject_data[s]["features"] for s in train_sids])
        scaler = StandardScaler()
        scaler.fit(train_X_flat)

        # Apply scaling to all subjects (in-memory; don't mutate subject_data)
        scaled_data: dict[str, dict] = {}
        for sid in np.concatenate([train_sids, test_sids]):
            raw_feats = subject_data[sid]["features"]
            scaled_data[sid] = {
                "features": scaler.transform(raw_feats),
                "ages": subject_data[sid]["ages"],
                "row_indices": subject_data[sid]["row_indices"],
            }

        # --- Step 2: fit LME on train subjects if needed ---
        lme_subjects_train = lme_slopes_train = None
        lme_subjects_full = lme_slopes_full = None
        if spec["needs_lme"]:
            # Build flat train matrix (already scaled)
            train_X_lme = np.vstack([scaled_data[s]["features"] for s in train_sids])
            train_ages_lme = np.concatenate([scaled_data[s]["ages"] for s in train_sids])
            train_sids_lme = np.concatenate([[s] * len(scaled_data[s]["ages"]) for s in train_sids])

            lme_est = make_lme_estimator(n_features, lme_threshold)
            lme_est.fit(train_X_lme, train_ages_lme, train_sids_lme)

            # Transform train subjects
            all_train_X = np.vstack([scaled_data[s]["features"] for s in train_sids])
            all_train_ages = np.concatenate([scaled_data[s]["ages"] for s in train_sids])
            all_train_sids = np.concatenate([[s] * len(scaled_data[s]["ages"]) for s in train_sids])
            lme_subjects_train, lme_slopes_train = lme_est.transform(all_train_X, all_train_ages, all_train_sids)

            # Transform test subjects
            all_test_X = np.vstack([scaled_data[s]["features"] for s in test_sids])
            all_test_ages = np.concatenate([scaled_data[s]["ages"] for s in test_sids])
            all_test_sids = np.concatenate([[s] * len(scaled_data[s]["ages"]) for s in test_sids])
            lme_subjects_test, lme_slopes_test = lme_est.transform(all_test_X, all_test_ages, all_test_sids)

            # Merge into combined arrays for build_agg_matrix
            lme_subjects_full = np.concatenate([lme_subjects_train, lme_subjects_test])
            lme_slopes_full = np.vstack([lme_slopes_train, lme_slopes_test])

        # --- Step 3: build aggregated feature matrices ---
        X_train, y_train = _build_agg_matrix(
            train_sids, scaled_data, df, aggregation, arm,
            lme_subjects=lme_subjects_full, lme_slopes=lme_slopes_full,
        )
        X_test, y_test = _build_agg_matrix(
            test_sids, scaled_data, df, aggregation, arm,
            lme_subjects=lme_subjects_full, lme_slopes=lme_slopes_full,
        )

        if len(y_train) < 2 or len(y_test) < 1:
            continue

        # --- Step 4: inner CV for Ridge alpha selection ---
        inner_groups = train_sids  # one row per subject in aggregated matrix
        best_alpha = _select_alpha(X_train, y_train, inner_groups, ridge_alphas, n_inner_folds)

        # --- Step 5: fit final Ridge and predict ---
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Step 6: metrics ---
        metrics = _regression_metrics(y_test, y_pred)
        # MAE-optimal constant predictor is the median (not mean).
        # For discrete targets this equals the mode; using mean would overestimate the baseline.
        chance_mae = float(np.mean(np.abs(y_test - np.median(y_train))))

        # Per-subject residuals: list of (subject_id, y_true, y_pred) for test subjects
        per_subject = [
            {"subject_id": str(sid), "y_true": float(yt), "y_pred": float(yp),
             "abs_error": float(abs(yt - yp))}
            for sid, yt, yp in zip(test_sids, y_test, y_pred)
        ]

        fold_results.append(
            {
                "fold": fold_idx,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
                "r": metrics["pearson"],
                "chance_mae": chance_mae,
                "best_alpha": best_alpha,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_subjects_train": len(train_sids),
                "n_subjects_test": len(test_sids),
                "aggregation": aggregation,
                "arm": arm,
                "band": band,
                "per_subject": per_subject,
            }
        )

    return fold_results


def _select_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    n_inner_folds: int,
) -> float:
    """Inner CV alpha selection using GroupKFold."""
    n_groups = len(np.unique(groups))
    if n_groups < n_inner_folds:
        # Not enough groups for inner CV; return middle alpha
        return float(alphas[len(alphas) // 2])

    gkf_inner = GroupKFold(n_splits=min(n_inner_folds, n_groups))
    gs = GridSearchCV(
        Ridge(),
        {"alpha": alphas},
        cv=gkf_inner,
        scoring="neg_mean_absolute_error",
        refit=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_train, y_train, groups=groups)
    return float(gs.best_params_["alpha"])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², Pearson r."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson = float(pearsonr(y_true, y_pred).statistic)
    else:
        pearson = float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}
