from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


METADATA_KEYS = [
    "id",
    "participant_id",
    "session",
    "session_type",
    "age",
    "sex",
    "timepoint",
    "directory",
    "days",
    "days_since_enrollment",
    "studies",
    "study_num",
    "image_path",
    "mask_path",
]


def _scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else value


def make_crosssectional_matrix(feature_npz):
    data = np.load(feature_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    meta = {}
    for key in METADATA_KEYS:
        if key in data.files:
            meta[key] = data[key]
    return X, pd.DataFrame(meta)


def make_longitudinal_matrix(feature_npz, pair_df, mode):
    data = np.load(feature_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    idx0 = pair_df["index_t0"].to_numpy(dtype=int)
    idx1 = pair_df["index_t1"].to_numpy(dtype=int)
    x0 = X[idx0]
    x1 = X[idx1]
    delta = x1 - x0

    if mode == "concat":
        X_pair = np.concatenate([x0, x1], axis=1)
    elif mode == "delta":
        X_pair = delta
    elif mode == "concat_delta":
        X_pair = np.concatenate([x0, x1, delta], axis=1)
    elif mode == "abs_delta":
        X_pair = np.abs(delta)
    elif mode == "annualized_delta":
        years = pair_df["delta_days"].to_numpy(dtype=np.float32) / 365.25
        years = np.where(np.abs(years) < 1e-8, np.nan, years)
        X_pair = delta / years[:, None]
    else:
        raise ValueError(f"Unsupported pair mode: {mode}")

    valid = np.isfinite(X_pair).all(axis=1)
    return X_pair[valid], pair_df.loc[valid].reset_index(drop=True)


def _split_groups(X, y, groups, seed, test_size):
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    return train_idx, test_idx


def _regression_metrics(y_true, y_pred):
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["pearson"] = float(pearsonr(y_true, y_pred).statistic)
        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["pearson"] = np.nan
        out["spearman"] = np.nan
    return out


def _classification_metrics(y_true, y_pred, y_score=None):
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["auc"] = np.nan
    return out


def _result_rows(
    metrics,
    feature_file,
    feature_meta,
    task,
    analysis_type,
    pair_mode,
    model,
    split_seed,
    train_idx,
    test_idx,
    groups,
):
    groups = np.asarray(groups)
    rows = []
    for metric, value in metrics.items():
        rows.append(
            {
                "feature_file": str(feature_file),
                "cnn_arch": feature_meta.get("cnn_arch", "double_conv"),
                "scaler": feature_meta.get("scaler", ""),
                "channels": feature_meta.get("channels", ""),
                "cnn_seed": feature_meta.get("seed", ""),
                "task": task,
                "analysis_type": analysis_type,
                "pair_mode": pair_mode,
                "model": model,
                "split_seed": split_seed,
                "metric": metric,
                "value": value,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_subjects_train": int(len(set(groups[train_idx]))),
                "n_subjects_test": int(len(set(groups[test_idx]))),
            }
        )
    return rows


def feature_metadata(feature_npz) -> dict:
    data = np.load(feature_npz, allow_pickle=True)
    meta = {
        "scaler": str(_scalar(data["scaler"])) if "scaler" in data.files else "",
        "channels": str(_scalar(data["channels"])) if "channels" in data.files else "",
        "seed": int(_scalar(data["seed"])) if "seed" in data.files else "",
    }
    if "cnn_arch" in data.files:
        meta["cnn_arch"] = str(_scalar(data["cnn_arch"]))
    else:
        # Infer from filename for older files that predate cnn_arch storage.
        stem = Path(feature_npz).stem
        parts = {seg.split("-", 1)[0]: seg.split("-", 1)[1] for seg in stem.split("__")[1:] if "-" in seg}
        meta["cnn_arch"] = parts.get("model", "double_conv")
    return meta


def evaluate_regression(
    X,
    y,
    groups,
    seeds,
    feature_file,
    feature_meta,
    task,
    analysis_type,
    pair_mode,
    test_size=0.1,
    rf_params=None,
):
    y = np.asarray(y, dtype=np.float32)
    groups = np.asarray(groups)
    valid = np.isfinite(y)
    X, y, groups = X[valid], y[valid], groups[valid]
    if len(np.unique(groups)) < 2 or len(y) < 3:
        return []

    rf_params = rf_params or {}
    rows = []
    for seed in seeds:
        train_idx, test_idx = _split_groups(X, y, groups, seed, test_size)
        models = {
            "rf": RandomForestRegressor(random_state=seed, n_jobs=1, **rf_params),
            "ridge_reg": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        }
        for name, model in models.items():
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            rows.extend(
                _result_rows(
                    _regression_metrics(y[test_idx], pred),
                    feature_file,
                    feature_meta,
                    task,
                    analysis_type,
                    pair_mode,
                    name,
                    seed,
                    train_idx,
                    test_idx,
                    groups,
                )
            )
    return rows


def evaluate_classification(
    X,
    y,
    groups,
    seeds,
    feature_file,
    feature_meta,
    task,
    analysis_type,
    pair_mode,
    test_size=0.1,
    rf_params=None,
):
    labels = pd.Series(y).astype(str)
    valid = labels.notna() & (labels != "") & (labels != "nan")
    X = X[valid.to_numpy()]
    groups = np.asarray(groups)[valid.to_numpy()]
    labels = labels[valid]
    if labels.nunique() < 2 or len(np.unique(groups)) < 2:
        return []

    encoded = LabelEncoder().fit_transform(labels)
    rf_params = rf_params or {}
    rows = []
    for seed in seeds:
        train_idx, test_idx = _split_groups(X, encoded, groups, seed, test_size)
        if len(np.unique(encoded[train_idx])) < 2 or len(np.unique(encoded[test_idx])) < 2:
            continue
        models = {
            "rf": RandomForestClassifier(
                random_state=seed,
                n_jobs=1,
                max_features="sqrt",
                class_weight="balanced",
                **rf_params,
            ),
            "logistic_clf": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=5000, class_weight="balanced"),
                    ),
                ]
            ),
        }
        for name, model in models.items():
            model.fit(X[train_idx], encoded[train_idx])
            pred = model.predict(X[test_idx])
            score = None
            if len(np.unique(encoded)) == 2:
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(X[test_idx])[:, 1]
                elif hasattr(model, "decision_function"):
                    score = model.decision_function(X[test_idx])
            rows.extend(
                _result_rows(
                    _classification_metrics(encoded[test_idx], pred, score),
                    feature_file,
                    feature_meta,
                    task,
                    analysis_type,
                    pair_mode,
                    name,
                    seed,
                    train_idx,
                    test_idx,
                    groups,
                )
            )
    return rows


def feature_file_id(path: str | Path) -> str:
    return Path(path).name
