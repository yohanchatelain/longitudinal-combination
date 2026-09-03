"""Map CNN feature importance to brain regions via gradient attribution.

Computes per-region importance comparable to FS ROI group-diff results
(features_fs_roi_<agg>.csv). For each aggregation type and each CNN seed:

  1. Build per-subject aggregated CNN feature vectors.
  2. Derive feature-level importance weight vector(s) w (Child vs Adult / PD vs
     HC), one per --weight-types entry: 'tstat' (Welch t-statistic, default),
     'shap' (signed mean TreeSHAP), 'hybrid' (RF permutation-importance x sign).
  3. Back-propagate each w through the frozen CNN for each sampled subject (one
     shared forward pass per subject) to get voxel-level attribution maps
     (signed and absolute).
  4. Project maps to brain regions via aparc+aseg atlas from FreeSurfer.
  5. Average across subjects and seeds; save CSVs and plots, tagged by weight type.

Usage:
    python -m brainage_agg.experiment.run_voxel_importance
    python -m brainage_agg.experiment.run_voxel_importance \\
        --dataset nki --arch double_conv --scaler minmax --channels t1_rank_sobel \\
        --n-subjects 50 --n-seeds 5 --agg mean annualized_rate --save-voxel-map
    python -m brainage_agg.experiment.run_voxel_importance --dataset ppmi --skip-lme
    python -m brainage_agg.experiment.run_voxel_importance \\
        --weight-types tstat shap hybrid --n-seeds 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import (
    backprop_from_w,
    forward_cache,
    load_freesurfer_lut,
    project_to_atlas,
    subject_parcellation_path,
)
from brainage_agg.analysis.importance_weights import (
    compute_w_hybrid,
    compute_w_shap,
    compute_w_t,
    fit_rf,
    rank_normalize,
)
from brainage_agg.analysis.permutation_null import (
    WelfordAccumulator,
    derive_threshold,
    null_correction,
)
from brainage_agg.validation.statistics import empirical_region_pvalues
from brainage_agg.validation.provenance import environment_snapshot, git_state, sha256_file
from brainage_agg.analysis.group_diff import (
    _get_dataset_config,
    build_aggregated_vectors,
)
from brainage_agg.agg.aggregations import AGGREGATION_SPECS
from brainage_agg.agg.lme import make_lme_estimator
from brainage_agg.data.manifest import load_manifest
from brainage_agg.features.loader import load_features, align_to_manifest
from src.io import load_cohort, load_mgz_float32, load_mgz_mask
from src.models import CNN3D_CovPool, CNN3D_DoubleConv, freeze_model, init_xavier
from src.preprocessing import build_channels
from src.utils import set_seed


# ---------------------------------------------------------------------------
# CNN model factory — mirrors build_model() in scripts/extract_all_features.py
# ---------------------------------------------------------------------------

def _build_cnn(arch: str, channels: list[str], seed: int, device: str) -> torch.nn.Module:
    set_seed(seed)
    if arch == "double_conv":
        model = CNN3D_DoubleConv(in_channels=len(channels), use_norm=True, activation="leaky_relu")
    elif arch == "cov_pool":
        model = CNN3D_CovPool(in_channels=len(channels))
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'double_conv' or 'cov_pool'.")
    init_xavier(model)
    freeze_model(model)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# NPZ file discovery
# ---------------------------------------------------------------------------

def _find_npz_files(feature_dir: Path, arch: str, scaler: str, channels: str) -> list[Path]:
    """Return all NPZ files matching (arch, scaler, channels) sorted by seed."""
    if arch == "double_conv":
        pattern = f"features__scaler-{scaler}__channels-{channels}__seed-*.npz"
    else:
        pattern = f"features__model-{arch}__scaler-{scaler}__channels-{channels}__seed-*.npz"
    return sorted(feature_dir.glob(pattern))


# ---------------------------------------------------------------------------
# Per-seed aggregated features and importance weights
# ---------------------------------------------------------------------------

def _build_aggregated_features(
    npz_path: Path,
    manifest: pd.DataFrame,
    aggregation: str,
    groups: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (X_agg, bands_agg, seed) for one extractor × aggregation.

    Returns None if there is insufficient data for this aggregation.
    """
    spec = AGGREGATION_SPECS[aggregation]
    X_npz, meta_df, meta = load_features(npz_path)
    seed = meta["seed"]
    subject_data = align_to_manifest(X_npz, meta_df, manifest)
    n_feat = X_npz.shape[1]

    lme_subjects = lme_slopes = None
    if spec["needs_lme"]:
        cohort_col = "cohort_span1" if "cohort_span1" in manifest.columns else "cohort_all"
        eligible = manifest[manifest[cohort_col] & manifest["band"].isin(list(groups))]
        flat_X, flat_ages, flat_sids = [], [], []
        for _, mrow in eligible.iterrows():
            sid = mrow["subject_id"]
            if sid not in subject_data:
                continue
            sd = subject_data[sid]
            for feat, age in zip(sd["features"], sd["ages"]):
                flat_X.append(feat)
                flat_ages.append(age)
                flat_sids.append(sid)
        if not flat_X:
            return None
        flat_X = np.array(flat_X, dtype=np.float64)
        flat_ages = np.array(flat_ages, dtype=np.float64)
        flat_sids = np.array(flat_sids)
        ages_c = flat_ages - flat_ages.mean()
        # Always use VectorizedOLS for CNN (thousands of features)
        lme_est = make_lme_estimator(n_feat, threshold=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lme_est.fit(flat_X, ages_c, flat_sids)
        lme_subjects, lme_slopes = lme_est.transform(flat_X, ages_c, flat_sids)

    X_agg, bands_agg, _ = build_aggregated_vectors(
        manifest, subject_data, aggregation,
        lme_subjects=lme_subjects, lme_slopes=lme_slopes,
        groups=groups,
    )
    if len(X_agg) < 6 or X_agg.shape[1] == 0:
        return None

    return X_agg, bands_agg, seed


def _compute_seed_weights(
    X_agg: np.ndarray,
    bands_agg: np.ndarray,
    groups: tuple[str, str],
    weight_types: list[str],
    seed: int,
    rf_n_estimators: int = 200,
    rf_permutation_repeats: int = 10,
    rf_min_samples_leaf: int = 5,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute raw (pre-standardization) importance weight vectors.

    One RandomForestClassifier (random_state=seed) backs both 'shap' and
    'hybrid' if either is requested. 'hybrid' diagnostics (oob_score,
    train_acc, imp_nonzero_frac) are printed (if verbose) for the Phase 1
    sanity check — a degenerate (~0) imp_nonzero_frac on real data signals
    an overfit RF. `verbose=False` suppresses this (used for the Phase 4b
    permutation-null refits, which call this N times per seed).
    """
    weights: dict[str, np.ndarray] = {}

    if "tstat" in weight_types:
        weights["tstat"] = compute_w_t(X_agg, bands_agg, groups)

    rf = None
    if "shap" in weight_types or "hybrid" in weight_types:
        rf = fit_rf(
            X_agg, bands_agg, groups, seed,
            n_estimators=rf_n_estimators, min_samples_leaf=rf_min_samples_leaf,
        )

    if "shap" in weight_types:
        weights["shap"] = compute_w_shap(rf, X_agg, groups)

    if "hybrid" in weight_types:
        w_hybrid, diagnostics = compute_w_hybrid(
            rf, X_agg, bands_agg, groups,
            n_repeats=rf_permutation_repeats, seed=seed,
        )
        weights["hybrid"] = w_hybrid
        if verbose:
            print()  # newline after the "Importance weights: ... " progress prefix
            print(
                f"      [hybrid seed={seed}] oob_score={diagnostics['oob_score']:.3f} "
                f"train_acc={diagnostics['train_acc']:.3f} "
                f"imp_nonzero_frac={diagnostics['imp_nonzero_frac']:.3f}",
                end=" ",
            )

    return weights


def _attribute_one_subject(
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    weights: dict[str, np.ndarray],
    weight_types: list[str],
    device: str,
    downsample_factor: int,
    null_weight_sets: list[dict[str, np.ndarray]] | None = None,
    allow_cpu_fallback: bool = True,
    null_map_callback: Callable[[int, str, np.ndarray, np.ndarray], None] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Run one shared forward pass, then backprop_from_w per weight type.

    Returns ``(observed, null)``:
      - ``observed[weight_type] = (signed_map, abs_map)`` — as before.
      - ``null[weight_type] = (null_mean_signed, null_var_signed,
        null_mean_abs, null_var_abs)`` — Welford mean/var across
        ``null_weight_sets`` (one full backprop per permutation per weight
        type). Empty dict if ``null_weight_sets`` is None/empty.

    On CUDA OOM (forward or backward), retries the entire subject on CPU —
    forward_cache moves `model` to the requested device itself, so no manual
    device bookkeeping is needed across calls.
    """
    n_model_feat = getattr(model, "out_features", None)
    cache = forward_cache(model, x_tensor, device=device, downsample_factor=downsample_factor)
    n_perm = len(null_weight_sets) if null_weight_sets else 0
    try:
        observed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, weight_type in enumerate(weight_types):
            is_last_observed = i == len(weight_types) - 1
            retain = not (is_last_observed and n_perm == 0)
            observed[weight_type] = backprop_from_w(
                cache, weights[weight_type], n_model_feat, retain_graph=retain
            )

        null: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        if n_perm:
            for wi, weight_type in enumerate(weight_types):
                acc_signed = WelfordAccumulator(cache.original_size)
                acc_abs = WelfordAccumulator(cache.original_size)
                for pi in range(n_perm):
                    is_last = (wi == len(weight_types) - 1) and (pi == n_perm - 1)
                    signed_map, abs_map = backprop_from_w(
                        cache, null_weight_sets[pi][weight_type], n_model_feat,
                        retain_graph=not is_last,
                    )
                    if null_map_callback is not None:
                        null_map_callback(pi, weight_type, signed_map, abs_map)
                    acc_signed.update(signed_map)
                    acc_abs.update(abs_map)
                mean_s, var_s = acc_signed.finalize()
                mean_a, var_a = acc_abs.finalize()
                null[weight_type] = (mean_s, var_s, mean_a, var_a)

        return observed, null
    except RuntimeError as exc:
        if (
            allow_cpu_fallback
            and "out of memory" in str(exc).lower()
            and "cuda" in str(device)
        ):
            torch.cuda.empty_cache()
            return _attribute_one_subject(
                model, x_tensor, weights, weight_types, "cpu", downsample_factor,
                null_weight_sets, allow_cpu_fallback,
                null_map_callback,
            )
        raise


# ---------------------------------------------------------------------------
# Subject sampling
# ---------------------------------------------------------------------------

def _sample_subjects(
    cohort: pd.DataFrame,
    groups: tuple[str, str],
    n_subjects: int,
    manifest: pd.DataFrame,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Sample up to n_subjects rows from cohort, one per subject, stratified by group.

    Takes the earliest session for each subject. Only subjects present in the
    manifest with a known band are included.
    """
    band_by_id = {str(k): v for k, v in manifest.set_index("subject_id")["band"].to_dict().items()}

    cohort = cohort.copy()
    cohort["_band"] = cohort["id"].astype(str).map(band_by_id)
    cohort = cohort[cohort["_band"].isin(list(groups))]
    # One row per subject — use the earliest timepoint
    cohort = (
        cohort.sort_values("timepoint")
        .groupby("id", sort=False)
        .first()
        .reset_index()
    )

    group_a, group_b = groups
    n_each = n_subjects // 2
    rng = np.random.default_rng(rng_seed)
    parts = []
    for grp in [group_a, group_b]:
        grp_rows = cohort[cohort["_band"] == grp].reset_index(drop=True)
        n = min(n_each, len(grp_rows))
        idx = rng.choice(len(grp_rows), size=n, replace=False)
        parts.append(grp_rows.iloc[idx])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Subject-level MRI loading helpers
# ---------------------------------------------------------------------------

def _preprocess_subject(
    row: pd.Series,
    scaler: str,
    channels: list[str],
) -> np.ndarray | None:
    """Load brain.mgz and preprocess to (C, H, W, D) float32, or None on failure."""
    try:
        img = load_mgz_float32(str(row["image_path"]))
        mask_path = str(row.get("mask_path", ""))
        if mask_path and Path(mask_path).exists():
            mask = load_mgz_mask(mask_path)
        else:
            mask = img > 0
        return build_channels(img, mask, scaler=scaler, channels=channels)
    except Exception as exc:
        print(f"    Warning: preprocessing failed for {row.get('directory', '?')}: {exc}")
        return None


def _load_parcellation(
    row: pd.Series,
    fs_root: str | None,
    parc_filename: str = "aparc+aseg.mgz",
) -> np.ndarray | None:
    """Load the requested parcellation file as int32, or None if unavailable."""
    if fs_root is None:
        return None
    parc_path = Path(fs_root) / str(row["directory"]) / "mri" / parc_filename
    if not parc_path.exists():
        return None
    try:
        return np.asarray(nib.load(str(parc_path)).dataobj, dtype=np.int32)
    except Exception as exc:
        print(f"    Warning: could not load parcellation for {row.get('directory', '?')}: {exc}")
        return None




def _preprocess_one_subject(
    row_dict: dict,
    scaler: str,
    channels: list[str],
    fs_root: str | None,
    parc_filename: str = "aparc+aseg.mgz",
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load, preprocess, and load parcellation for one subject.

    Returns (arr, parc, affine) — any element may be None on failure.
    Designed to be called from a joblib worker process.
    """
    row = pd.Series(row_dict)
    arr = _preprocess_subject(row, scaler, channels)
    parc = _load_parcellation(row, fs_root, parc_filename)
    affine = None
    try:
        affine = nib.load(str(row["image_path"])).affine
    except Exception:
        pass
    return arr, parc, affine


def _preprocess_all_subjects(
    sampled: pd.DataFrame,
    scaler: str,
    channels: list[str],
    fs_root: str | None,
    n_jobs: int = 1,
    parc_filename: str = "aparc+aseg.mgz",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Preprocess all sampled subjects in parallel, returning only valid entries.

    Results are cached in RAM and reused across all aggregations and seeds,
    eliminating the O(n_agg × n_seeds × n_subjects) preprocessing that otherwise
    dominates runtime (rank channel: ~3 min/subject × 50 subjects × 10 seeds = 25 h).

    Memory: 50 subjects × 3ch × 256³ × float32 ≈ 10 GB  (parcellations: +3 GB).
    """
    from joblib import Parallel, delayed

    rows = [row.to_dict() for _, row in sampled.iterrows()]
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(_preprocess_one_subject)(row_dict, scaler, channels, fs_root, parc_filename)
        for row_dict in rows
    )

    valid = [
        (arr, parc, affine)
        for arr, parc, affine in results
        if arr is not None and parc is not None
    ]
    return valid


# ---------------------------------------------------------------------------
# FS ROI comparison helpers
# ---------------------------------------------------------------------------

def _find_matching_roi_features(region_name: str, roi_df: pd.DataFrame) -> pd.DataFrame:
    """Match a DK LUT region name to FS ROI feature rows.

    Desikan-Killiany cortical (aparc+aseg):
      ctx-lh-superiorfrontal  → features starting with 'lh_superiorfrontal_'
      ctx-rh-superiorfrontal  → features starting with 'rh_superiorfrontal_'
    Subcortical: Left-Hippocampus → feature 'Left-Hippocampus' (direct match)
    """
    direct = roi_df[roi_df["feature"] == region_name]
    if not direct.empty:
        return direct

    if region_name.startswith("ctx-lh-"):
        prefix = "lh_" + region_name[7:] + "_"
        return roi_df[roi_df["feature"].str.startswith(prefix)]
    if region_name.startswith("ctx-rh-"):
        prefix = "rh_" + region_name[7:] + "_"
        return roi_df[roi_df["feature"].str.startswith(prefix)]

    return pd.DataFrame()


def _load_roi_comparison(result_dir: Path, lut: dict[int, str]) -> pd.DataFrame | None:
    """Load FS ROI group-diff results and aggregate max |cohen_d| per region.

    Tries option3_raw_baseline_fs_roi.csv first, then features_fs_roi_mean.csv.
    Returns a DataFrame with columns [label_id, region_name, max_abs_cohen_d_roi],
    or None if no FS ROI results are found.
    """
    gd_dir = result_dir / "group_diff"
    for fname in ("option3_raw_baseline_fs_roi.csv", "features_fs_roi_mean.csv"):
        roi_path = gd_dir / fname
        if roi_path.exists():
            break
    else:
        return None

    roi_df = pd.read_csv(roi_path)
    if "feature" not in roi_df.columns or "cohen_d" not in roi_df.columns:
        return None

    records = []
    for label_id, region_name in lut.items():
        matching = _find_matching_roi_features(region_name, roi_df)
        if matching.empty:
            continue
        records.append({
            "label_id": label_id,
            "region_name": region_name,
            "max_abs_cohen_d_roi": float(matching["cohen_d"].abs().max()),
        })
    return pd.DataFrame(records) if records else None


# ---------------------------------------------------------------------------
# Region importance DataFrame builder
# ---------------------------------------------------------------------------

def _accumulate_region_maps(
    signed_map: np.ndarray,
    abs_map: np.ndarray,
    parc: np.ndarray,
    signed_accum: dict[int, list[float]],
    abs_accum: dict[int, list[float]],
    nvox_accum: dict[int, int] | None = None,
) -> None:
    """Project (signed_map, abs_map) to atlas regions and accumulate per-region means."""
    reg_signed, reg_abs = project_to_atlas(signed_map, abs_map, parc)
    for lid, (val, n_vox) in reg_abs.items():
        abs_accum.setdefault(lid, []).append(val)
        if nvox_accum is not None:
            nvox_accum[lid] = n_vox
    for lid, (val, _) in reg_signed.items():
        signed_accum.setdefault(lid, []).append(val)


def _region_importance_df(
    region_signed_accum: dict[int, list[float]],
    region_abs_accum: dict[int, list[float]],
    region_nvox: dict[int, int],
    lut: dict[int, str],
    groups: tuple[str, str],
    extra_accums: dict[str, dict[int, list[float]]] | None = None,
) -> pd.DataFrame:
    """Build the per-region importance table.

    `extra_accums` (Phase 4b null-correction): optional `{column_name:
    {label_id: [values...]}}` — each is averaged per region and added as an
    extra column (e.g. null mean/variance, corrected, and z-score maps).
    """
    group_a, group_b = groups
    rows = []
    for label_id in sorted(region_abs_accum.keys()):
        region_name = lut.get(label_id, f"label_{label_id}")
        mean_abs = float(np.mean(region_abs_accum[label_id]))
        mean_signed = float(np.mean(region_signed_accum.get(label_id, [0.0])))
        n_vox = region_nvox.get(label_id, 0)

        if region_name.startswith(("Left-", "ctx-lh-", "wm-lh-")):
            hemisphere = "left"
        elif region_name.startswith(("Right-", "ctx-rh-", "wm-rh-")):
            hemisphere = "right"
        else:
            hemisphere = "bilateral"

        row = {
            "label_id": label_id,
            "region_name": region_name,
            "hemisphere": hemisphere,
            "mean_abs_importance": mean_abs,
            "mean_signed_importance": mean_signed,
            "direction": f"{group_a} > {group_b}" if mean_signed > 0 else f"{group_b} > {group_a}",
            "n_voxels": n_vox,
        }
        if extra_accums:
            for col_name, accum in extra_accums.items():
                row[col_name] = float(np.mean(accum.get(label_id, [0.0])))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_barplot(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 30,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return

    group_a, group_b = groups
    df_plot = df.nlargest(top_n, "mean_abs_importance").sort_values(
        "mean_abs_importance", ascending=True
    )
    colors = [
        "tomato" if s > 0 else "steelblue"
        for s in df_plot["mean_signed_importance"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.28)))
    ax.barh(df_plot["region_name"], df_plot["mean_abs_importance"], color=colors)
    legend_elements = [
        Patch(facecolor="tomato",    label=f"{group_a} > {group_b}"),
        Patch(facecolor="steelblue", label=f"{group_b} > {group_a}"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    ax.set_xlabel("Mean |attribution|")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path.name}")


def _make_scatter(
    merged: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        return

    if merged.empty:
        return

    x = merged["mean_abs_importance"].values
    y = merged["max_abs_cohen_d_roi"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.6)
    for _, row in merged.iterrows():
        ax.annotate(
            row["region_name"],
            (row["mean_abs_importance"], row["max_abs_cohen_d_roi"]),
            fontsize=4, ha="center", va="bottom",
        )

    if len(merged) >= 3:
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        ax.set_title(f"{title}\nPearson r={r:.2f}, Spearman ρ={rho:.2f}")
    else:
        ax.set_title(title)

    ax.set_xlabel("CNN mean |attribution|")
    ax.set_ylabel("FS ROI max |Cohen's d|")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    project_root: Path,
    dataset: str,
    arch: str,
    scaler: str,
    channels: str,
    n_subjects: int,
    n_seeds: int | None,
    aggregations: list[str],
    save_voxel_map: bool,
    device: str,
    skip_lme: bool,
    downsample_factor: int = 2,
    weight_types: list[str] | None = None,
    rf_n_estimators: int = 200,
    rf_permutation_repeats: int = 10,
    rf_min_samples_leaf: int = 5,
    n_permutations: int = 0,
    null_rng_seed: int = 0,
    strict_complete: bool = False,
    validation_output_dir: Path | None = None,
) -> None:
    weight_types = weight_types or ["tstat"]
    ds_cfg = _get_dataset_config(dataset, project_root)
    groups: tuple[str, str] = ds_cfg["groups"]
    result_dir = Path(ds_cfg["result_dir"])
    feature_dir = Path(ds_cfg["feature_dir"])
    group_a, group_b = groups

    # --- Manifest ---
    manifest_path = result_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run brainage_agg.experiment.run first to build the manifest."
        )
    manifest = load_manifest(manifest_path)

    # --- Find CNN NPZ files ---
    npz_files = _find_npz_files(feature_dir, arch, scaler, channels)
    if not npz_files:
        raise FileNotFoundError(
            f"No NPZ files found for arch={arch}, scaler={scaler}, channels={channels} "
            f"in {feature_dir}"
        )
    if n_seeds is not None:
        if strict_complete and len(npz_files) < n_seeds:
            raise RuntimeError(f"Requested {n_seeds} seeds but found only {len(npz_files)}")
        npz_files = npz_files[:n_seeds]
    print(
        f"Found {len(npz_files)} seed(s) for "
        f"arch={arch}, scaler={scaler}, channels={channels}"
    )

    # --- Load brainage_agg config for fs_root and cohort_csv ---
    if dataset == "ppmi":
        cfg_path = project_root / "brainage_agg" / "ppmi_config.yaml"
    else:
        cfg_path = project_root / "brainage_agg" / "config.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    cohort_csv = project_root / cfg["data"]["cohort_csv"]
    fs_root_cfg = cfg["data"].get("fs_root")
    fs_root = str(project_root / fs_root_cfg) if fs_root_cfg else None

    # --- Load cohort (with image paths for gradient computation) ---
    if fs_root is not None:
        cohort = load_cohort(str(cohort_csv), freesurfer_root=fs_root)
    else:
        # Dataset uses direct CSV image paths (e.g. PPMI NIfTI images)
        cohort = load_cohort(str(cohort_csv), use_csv_paths=True)
    if cohort.empty:
        print("No subjects with valid image paths found — cannot compute gradient attribution.")
        return

    # Parcellations require fs_root; warn early if absent
    if fs_root is None:
        print(
            "Warning: 'fs_root' is not set in the dataset config. "
            "FreeSurfer aparc+aseg parcellations are required for region projection. "
            "Voxel maps will be computed but region-level output will be empty."
        )

    # --- Sample subjects ---
    sampled = _sample_subjects(cohort, groups, n_subjects, manifest)
    if strict_complete and len(sampled) != n_subjects:
        raise RuntimeError(f"Requested {n_subjects} balanced subjects but sampled {len(sampled)}")
    print(
        f"Sampled {len(sampled)} subjects "
        f"({(sampled['_band'] == group_a).sum()} {group_a}, "
        f"{(sampled['_band'] == group_b).sum()} {group_b})"
    )

    # --- FreeSurfer LUT ---
    lut = load_freesurfer_lut(project_root / "FS-LUT.txt")

    # --- FS ROI comparison data (loaded once) ---
    roi_comparison = _load_roi_comparison(result_dir, lut)
    if roi_comparison is not None:
        print(f"Loaded FS ROI comparison data ({len(roi_comparison)} regions)")
    else:
        print(
            "No FS ROI group-diff results found — skipping CNN vs ROI comparison. "
            "Run brainage_agg.analysis.group_diff first."
        )

    parc_filename = "aparc+aseg.mgz"  # Desikan-Killiany for both NKI and PPMI
    print(f"Using parcellation: {parc_filename}")

    channels_list = channels.split("_")  # 't1_rank_sobel' → ['t1', 'rank', 'sobel']

    # --- Preprocess all subjects ONCE (cached across aggregations and seeds) ---
    # Without caching, preprocessing is repeated n_aggregations × n_seeds × n_subjects times.
    # The rank channel alone takes ~3 min/subject; at 50 subjects × 10 seeds × 6 aggs = 150 h.
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(
        f"\nPreprocessing {len(sampled)} subjects once "
        f"(parallel, n_jobs={n_jobs}, channels={channels})..."
    )
    t0 = time.time()
    cached_subjects = _preprocess_all_subjects(
        sampled, scaler, channels_list, fs_root, n_jobs, parc_filename
    )
    print(
        f"Done in {time.time() - t0:.0f}s — "
        f"{len(cached_subjects)}/{len(sampled)} subjects valid (arr + parcellation)"
    )
    if not cached_subjects:
        raise RuntimeError("No valid subjects after preprocessing")
    if strict_complete and len(cached_subjects) != len(sampled):
        raise RuntimeError(
            f"Preprocessing produced {len(cached_subjects)}/{len(sampled)} requested subjects"
        )

    # Grab affine from first valid subject for NIfTI export
    first_affine = next((aff for _, _, aff in cached_subjects if aff is not None), None)

    # Pre-build tensors once (avoids repeated tensor construction inside the seed loop).
    # brain_mask derived from preprocessed channels: build_channels zeros non-brain voxels,
    # so any(axis=0) reconstructs the effective per-subject brain mask without reloading files.
    cached_tensors = [
        (torch.tensor(arr[np.newaxis], dtype=torch.float32), parc, affine, arr.any(axis=0))
        for arr, parc, affine in cached_subjects
    ]
    del cached_subjects  # free numpy arrays; tensors hold the data

    # --- Process each aggregation ---
    for aggregation in aggregations:
        spec = AGGREGATION_SPECS[aggregation]
        if spec["needs_lme"] and skip_lme:
            print(f"\nSkipping {aggregation} (--skip-lme passed)")
            continue

        print(f"\n{'='*60}")
        print(f"Aggregation: {aggregation}")
        print(f"{'='*60}")

        out_dir = result_dir / "group_diff" / "voxel_importance" / aggregation
        if validation_output_dir is not None:
            out_dir = Path(validation_output_dir) / dataset / arch / aggregation
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Pass A: compute per-seed raw importance weights ---
        seed_weights: list[tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
        for npz_path in npz_files:
            print(f"  Importance weights: {npz_path.name} ...", end=" ", flush=True)
            agg_result = _build_aggregated_features(npz_path, manifest, aggregation, groups)
            if agg_result is None:
                if strict_complete:
                    raise RuntimeError(
                        f"Insufficient data for requested seed file {npz_path.name} "
                        f"and aggregation {aggregation}"
                    )
                print("skipped (insufficient data)")
                continue
            X_agg, bands_agg, seed = agg_result
            weights = _compute_seed_weights(
                X_agg, bands_agg, groups, weight_types, seed,
                rf_n_estimators=rf_n_estimators,
                rf_permutation_repeats=rf_permutation_repeats,
                rf_min_samples_leaf=rf_min_samples_leaf,
            )
            missing_weights = set(weight_types) - set(weights)
            if missing_weights:
                raise RuntimeError(f"Missing requested weight types: {sorted(missing_weights)}")
            seed_weights.append((seed, X_agg, bands_agg, weights))
            n_feat = next(iter(weights.values())).shape[0]
            print(f"done  n_feat={n_feat}")

        if not seed_weights:
            print(f"  No valid importance weights for '{aggregation}' — skipping.")
            continue

        # Standardize: rank-normalize each raw weight vector to a common
        # marginal so w_t, w_shap, w_hybrid are on equal footing before
        # backprop (their raw scales differ by orders of magnitude).
        for _, _, _, weights in seed_weights:
            for weight_type in weight_types:
                weights[weight_type] = rank_normalize(weights[weight_type])

        # --- Pass A': permutation-null weight sets (Phase 4b) ---
        # group_perm architecture-bias null, applied to ALL requested weight
        # types: for each seed, n_permutations group-label permutations, each
        # re-deriving {tstat, shap, hybrid} from scratch (refitting the RF for
        # shap/hybrid) and rank-normalizing. Expensive — disabled by default
        # (n_permutations=0).
        null_weights: dict[int, list[dict[str, np.ndarray]]] = {}
        if n_permutations > 0:
            print(
                f"\n  Computing {n_permutations} group-label-permutation null "
                f"weight set(s) per seed for {weight_types} ..."
            )
            for seed, X_agg, bands_agg, _ in seed_weights:
                rng = np.random.default_rng(null_rng_seed + seed)
                t0 = time.time()
                perms: list[dict[str, np.ndarray]] = []
                for _ in range(n_permutations):
                    perm_bands = rng.permutation(bands_agg)
                    w_perm = _compute_seed_weights(
                        X_agg, perm_bands, groups, weight_types, seed,
                        rf_n_estimators=rf_n_estimators,
                        rf_permutation_repeats=rf_permutation_repeats,
                        rf_min_samples_leaf=rf_min_samples_leaf,
                        verbose=False,
                    )
                    for weight_type in weight_types:
                        w_perm[weight_type] = rank_normalize(w_perm[weight_type])
                    perms.append(w_perm)
                null_weights[seed] = perms
                print(
                    f"    seed={seed}: {n_permutations} null weight set(s) "
                    f"in {time.time() - t0:.0f}s"
                )

        # --- Pass B: accumulate attribution maps across seeds and subjects ---
        # Preprocessing is already done (cached_tensors); only gradient computation here.
        # forward_cache runs once per subject; backprop_from_w runs once per weight type
        # on the same cached forward pass (plus n_permutations null backprops per
        # weight type if Phase 4b null-correction is enabled).
        sum_signed_by_seed: dict[str, list[np.ndarray]] = {wt: [] for wt in weight_types}
        sum_abs_by_seed:    dict[str, list[np.ndarray]] = {wt: [] for wt in weight_types}
        region_abs_accum:    dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_signed_accum: dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_nvox:         dict[str, dict[int, int]] = {wt: {} for wt in weight_types}

        # Phase 4b: group_perm null-correction accumulators (populated only if
        # n_permutations > 0). For each subject: null mean/var maps (architecture
        # bias), corrected = observed - null_mean, and z = (observed - null_mean) /
        # sqrt(null_var) — each projected to atlas regions and averaged like the
        # observed maps.
        region_null_signed_accum:      dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_null_abs_accum:         dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_corrected_signed_accum: dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_corrected_abs_accum:    dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_z_signed_accum:         dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}
        region_z_abs_accum:            dict[str, dict[int, list[float]]] = {wt: {} for wt in weight_types}

        # Confirmatory regional null: retain every permutation's regional
        # statistic. Keys are (seed, permutation_index, label_id); values are
        # subject-level region means, averaged only when materializing output.
        permutation_region_abs: dict[str, dict[tuple[int, int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        permutation_region_signed: dict[str, dict[tuple[int, int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        observed_region_abs_by_seed: dict[str, dict[tuple[int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        observed_region_signed_by_seed: dict[str, dict[tuple[int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        corrected_region_abs_by_seed: dict[str, dict[tuple[int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        z_region_abs_by_seed: dict[str, dict[tuple[int, int], list[float]]] = {
            wt: {} for wt in weight_types
        }
        expected_region_subjects: dict[int, int] = {}
        for _, subject_parcellation, _, _ in cached_tensors:
            for label_id in np.unique(subject_parcellation):
                if int(label_id) != 0:
                    expected_region_subjects[int(label_id)] = expected_region_subjects.get(int(label_id), 0) + 1

        n_maps_total = 0

        for seed, _, _, weights in seed_weights:
            null_suffix = f" x {n_permutations} null permutation(s)" if n_permutations > 0 else ""
            print(
                f"  Seed {seed}: computing {len(cached_tensors)} attribution map(s) "
                f"x {len(weight_types)} weight type(s){null_suffix} ...",
                flush=True,
            )
            t_seed = time.time()

            model = _build_cnn(arch, channels_list, seed, device)
            seed_null_weights = null_weights.get(seed)

            per_seed_signed: dict[str, list[np.ndarray]] = {wt: [] for wt in weight_types}
            per_seed_abs:    dict[str, list[np.ndarray]] = {wt: [] for wt in weight_types}

            for x_tensor, parc, _, brain_mask in cached_tensors:
                def retain_permutation_region(
                    permutation_index: int,
                    weight_type: str,
                    signed_map: np.ndarray,
                    abs_map: np.ndarray,
                ) -> None:
                    signed_copy = signed_map.copy()
                    abs_copy = abs_map.copy()
                    signed_copy[~brain_mask] = 0.0
                    abs_copy[~brain_mask] = 0.0
                    regional_signed, regional_abs = project_to_atlas(signed_copy, abs_copy, parc)
                    for label_id, (value, _) in regional_abs.items():
                        permutation_region_abs[weight_type].setdefault(
                            (seed, permutation_index, label_id), []
                        ).append(value)
                    for label_id, (value, _) in regional_signed.items():
                        permutation_region_signed[weight_type].setdefault(
                            (seed, permutation_index, label_id), []
                        ).append(value)

                try:
                    results, null_results = _attribute_one_subject(
                        model, x_tensor, weights, weight_types, device, downsample_factor,
                        null_weight_sets=seed_null_weights,
                        allow_cpu_fallback=not strict_complete,
                        null_map_callback=(retain_permutation_region if n_permutations > 0 else None),
                    )
                except RuntimeError as exc:
                    if strict_complete:
                        raise RuntimeError(
                            f"Gradient computation failed for seed={seed}: {exc}"
                        ) from exc
                    print(f"    Gradient computation failed: {exc}")
                    continue

                for weight_type, (signed_map, abs_map) in results.items():
                    signed_map[~brain_mask] = 0.0
                    abs_map[~brain_mask] = 0.0

                    per_seed_signed[weight_type].append(signed_map)
                    per_seed_abs[weight_type].append(abs_map)

                    _accumulate_region_maps(
                        signed_map, abs_map, parc,
                        region_signed_accum[weight_type], region_abs_accum[weight_type],
                        region_nvox[weight_type],
                    )
                    current_signed, current_abs = project_to_atlas(signed_map, abs_map, parc)
                    for label_id, (value, _) in current_abs.items():
                        observed_region_abs_by_seed[weight_type].setdefault(
                            (seed, label_id), []
                        ).append(value)
                    for label_id, (value, _) in current_signed.items():
                        observed_region_signed_by_seed[weight_type].setdefault(
                            (seed, label_id), []
                        ).append(value)

                    if weight_type in null_results:
                        null_mean_s, null_var_s, null_mean_a, null_var_a = null_results[weight_type]
                        null_mean_s = null_mean_s.copy()
                        null_mean_a = null_mean_a.copy()
                        null_mean_s[~brain_mask] = 0.0
                        null_mean_a[~brain_mask] = 0.0

                        corrected_signed = signed_map - null_mean_s
                        corrected_abs = abs_map - null_mean_a
                        z_signed = null_correction(signed_map, null_mean_s, null_var_s)
                        z_abs = null_correction(abs_map, null_mean_a, null_var_a)
                        for arr in (corrected_signed, corrected_abs, z_signed, z_abs):
                            arr[~brain_mask] = 0.0

                        _accumulate_region_maps(
                            null_mean_s, null_mean_a, parc,
                            region_null_signed_accum[weight_type], region_null_abs_accum[weight_type],
                        )
                        _accumulate_region_maps(
                            corrected_signed, corrected_abs, parc,
                            region_corrected_signed_accum[weight_type], region_corrected_abs_accum[weight_type],
                        )
                        _accumulate_region_maps(
                            z_signed, z_abs, parc,
                            region_z_signed_accum[weight_type], region_z_abs_accum[weight_type],
                        )
                        _, corrected_regions_abs = project_to_atlas(
                            corrected_signed, corrected_abs, parc
                        )
                        _, z_regions_abs = project_to_atlas(z_signed, z_abs, parc)
                        for label_id, (value, _) in corrected_regions_abs.items():
                            corrected_region_abs_by_seed[weight_type].setdefault(
                                (seed, label_id), []
                            ).append(value)
                        for label_id, (value, _) in z_regions_abs.items():
                            z_region_abs_by_seed[weight_type].setdefault(
                                (seed, label_id), []
                            ).append(value)

                n_maps_total += 1

            del model  # free GPU/CPU memory before next seed

            for weight_type in weight_types:
                if per_seed_signed[weight_type]:
                    sum_signed_by_seed[weight_type].append(np.mean(per_seed_signed[weight_type], axis=0))
                    sum_abs_by_seed[weight_type].append(np.mean(per_seed_abs[weight_type], axis=0))

            print(
                f"    {len(cached_tensors)} subject(s) in {time.time() - t_seed:.0f}s "
                f"({(time.time() - t_seed) / max(1, len(cached_tensors)):.1f}s/subject)"
            )

        if n_maps_total == 0:
            print(f"  No attribution maps produced for '{aggregation}' — skipping.")
            continue

        # --- Per-weight-type outputs ---
        for weight_type in weight_types:
            if not sum_signed_by_seed[weight_type]:
                print(f"  [{weight_type}] No attribution maps produced — skipping.")
                continue

            n_valid_seeds = len(sum_signed_by_seed[weight_type])
            mean_signed_global = np.mean(sum_signed_by_seed[weight_type], axis=0).astype(np.float32)
            mean_abs_global    = np.mean(sum_abs_by_seed[weight_type],    axis=0).astype(np.float32)
            print(
                f"  [{weight_type}] Aggregated maps from {n_valid_seeds} seed(s); "
                f"global max |attribution| = {mean_abs_global.max():.4g}"
            )

            extractor_key = (
                f"{arch}__{scaler}__{channels}__{weight_type}__nsub{n_subjects}__{n_valid_seeds}seeds"
            )

            # --- Save voxel maps as NIfTI ---
            if save_voxel_map and first_affine is not None:
                for tag, data in [("signed", mean_signed_global), ("abs", mean_abs_global)]:
                    nii_path = out_dir / f"voxel_map_{tag}_{extractor_key}.nii.gz"
                    nib.save(nib.Nifti1Image(data, affine=first_affine), str(nii_path))
                    print(f"  Saved {nii_path.name}")
            elif save_voxel_map:
                print("  Warning: could not save NIfTI (no subject affine captured)")

            # --- Phase 4b: null-correction extra columns + FWER threshold ---
            extra_accums = None
            if n_permutations > 0 and region_null_signed_accum[weight_type]:
                extra_accums = {
                    "mean_null_signed_importance": region_null_signed_accum[weight_type],
                    "mean_null_abs_importance": region_null_abs_accum[weight_type],
                    "mean_corrected_signed_importance": region_corrected_signed_accum[weight_type],
                    "mean_corrected_abs_importance": region_corrected_abs_accum[weight_type],
                    "mean_z_signed": region_z_signed_accum[weight_type],
                    "mean_z_abs": region_z_abs_accum[weight_type],
                }
                z_threshold = derive_threshold(
                    np.empty(mean_abs_global.shape, dtype=np.float32), alpha=0.05
                )
                print(
                    f"  [{weight_type}] Null-corrected ({n_permutations} group_perm "
                    f"permutation(s)); voxelwise |z| FWER threshold (alpha=0.05, "
                    f"Bonferroni over {mean_abs_global.size} voxels) = {z_threshold:.2f}"
                )

            # --- Region importance DataFrame ---
            region_df = _region_importance_df(
                region_signed_accum[weight_type], region_abs_accum[weight_type],
                region_nvox[weight_type], lut, groups,
                extra_accums=extra_accums,
            )
            csv_path = out_dir / f"region_importance_{extractor_key}.csv"
            region_df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path.name} ({len(region_df)} regions)")

            if n_permutations > 0:
                label_ids = np.array(sorted(region_abs_accum[weight_type]), dtype=np.int32)
                seeds_used = np.array([seed for seed, *_ in seed_weights], dtype=np.int32)
                permutation_abs = np.empty(
                    (len(seeds_used), n_permutations, len(label_ids)), dtype=np.float32
                )
                permutation_signed = np.empty_like(permutation_abs)
                for seed_index, current_seed in enumerate(seeds_used):
                    for permutation_index in range(n_permutations):
                        for label_index, label_id in enumerate(label_ids):
                            abs_values = permutation_region_abs[weight_type].get(
                                (int(current_seed), permutation_index, int(label_id)), []
                            )
                            signed_values = permutation_region_signed[weight_type].get(
                                (int(current_seed), permutation_index, int(label_id)), []
                            )
                            if strict_complete and (
                                len(abs_values) != expected_region_subjects.get(int(label_id), 0)
                                or len(signed_values) != expected_region_subjects.get(int(label_id), 0)
                            ):
                                raise RuntimeError(
                                    "Missing permutation-region statistic for "
                                    f"seed={current_seed}, permutation={permutation_index}, "
                                    f"label={label_id}, weight={weight_type}"
                                )
                            permutation_abs[seed_index, permutation_index, label_index] = np.mean(abs_values)
                            permutation_signed[seed_index, permutation_index, label_index] = np.mean(signed_values)
                observed_abs = np.array(
                    [np.mean(region_abs_accum[weight_type][int(label)]) for label in label_ids],
                    dtype=np.float32,
                )
                observed_signed = np.array(
                    [np.mean(region_signed_accum[weight_type][int(label)]) for label in label_ids],
                    dtype=np.float32,
                )
                observed_abs_by_seed = np.empty(
                    (len(seeds_used), len(label_ids)), dtype=np.float32
                )
                observed_signed_by_seed = np.empty_like(observed_abs_by_seed)
                corrected_abs_by_seed = np.empty_like(observed_abs_by_seed)
                z_abs_by_seed = np.empty_like(observed_abs_by_seed)
                for seed_index, current_seed in enumerate(seeds_used):
                    for label_index, label_id in enumerate(label_ids):
                        seed_abs = observed_region_abs_by_seed[weight_type].get(
                            (int(current_seed), int(label_id)), []
                        )
                        seed_signed = observed_region_signed_by_seed[weight_type].get(
                            (int(current_seed), int(label_id)), []
                        )
                        if strict_complete and (
                            len(seed_abs) != expected_region_subjects.get(int(label_id), 0)
                            or len(seed_signed) != expected_region_subjects.get(int(label_id), 0)
                        ):
                            raise RuntimeError(
                                f"Missing observed region statistic for seed={current_seed}, "
                                f"label={label_id}, weight={weight_type}"
                            )
                        observed_abs_by_seed[seed_index, label_index] = np.mean(seed_abs)
                        observed_signed_by_seed[seed_index, label_index] = np.mean(seed_signed)
                        corrected_values = corrected_region_abs_by_seed[weight_type].get(
                            (int(current_seed), int(label_id)), []
                        )
                        z_values = z_region_abs_by_seed[weight_type].get(
                            (int(current_seed), int(label_id)), []
                        )
                        if strict_complete and (
                            len(corrected_values) != expected_region_subjects.get(int(label_id), 0)
                            or len(z_values) != expected_region_subjects.get(int(label_id), 0)
                        ):
                            raise RuntimeError(
                                f"Missing corrected region statistic for seed={current_seed}, "
                                f"label={label_id}, weight={weight_type}"
                            )
                        corrected_abs_by_seed[seed_index, label_index] = np.mean(corrected_values)
                        z_abs_by_seed[seed_index, label_index] = np.mean(z_values)
                # Average matched permutation indices across seeds; the seed
                # dimension is retained in the NPZ for seed-bootstrap analysis.
                pointwise_p, maxT_p = empirical_region_pvalues(
                    observed_abs, permutation_abs.mean(axis=0), alternative="greater"
                )
                p_by_label = {int(label): float(p) for label, p in zip(label_ids, pointwise_p)}
                maxT_by_label = {int(label): float(p) for label, p in zip(label_ids, maxT_p)}
                region_df["empirical_p_one_sided"] = region_df["label_id"].map(p_by_label)
                region_df["maxT_fwer_p"] = region_df["label_id"].map(maxT_by_label)
                region_df.to_csv(csv_path, index=False)
                permutation_path = out_dir / f"permutation_region_stats_{extractor_key}.npz"
                np.savez_compressed(
                    permutation_path,
                    label_ids=label_ids,
                    seeds=seeds_used,
                    observed_abs=observed_abs,
                    observed_signed=observed_signed,
                    observed_abs_by_seed=observed_abs_by_seed,
                    observed_signed_by_seed=observed_signed_by_seed,
                    corrected_abs_by_seed=corrected_abs_by_seed,
                    empirical_standardized_abs_by_seed=z_abs_by_seed,
                    permutation_abs=permutation_abs,
                    permutation_signed=permutation_signed,
                    pointwise_p_one_sided=pointwise_p,
                    maxT_fwer_p=maxT_p,
                )
                print(f"  Saved {permutation_path.name} (permutation-indexed regional null)")

            # --- Bar plot ---
            _make_barplot(
                region_df,
                title=(
                    f"CNN voxel importance — {aggregation} ({weight_type})\n"
                    f"({arch}/{scaler}/{channels}, {n_valid_seeds} seeds, "
                    f"{group_a} vs {group_b})"
                ),
                out_path=out_dir / f"region_barplot_{extractor_key}.png",
                groups=groups,
            )

            # --- FS ROI comparison ---
            if roi_comparison is not None:
                merged = pd.merge(
                    region_df[["label_id", "region_name", "mean_abs_importance", "mean_signed_importance"]],
                    roi_comparison[["label_id", "region_name", "max_abs_cohen_d_roi"]],
                    on=["label_id", "region_name"],
                    how="inner",
                )
                if not merged.empty:
                    cmp_path = out_dir / f"cnn_vs_roi_comparison_{extractor_key}.csv"
                    merged.to_csv(cmp_path, index=False)
                    print(f"  Saved {cmp_path.name} ({len(merged)} matched regions)")
                    _make_scatter(
                        merged,
                        title=f"CNN vs FS ROI — {aggregation} ({weight_type})",
                        out_path=out_dir / f"cnn_vs_roi_scatter_{extractor_key}.png",
                    )

        if validation_output_dir is not None:
            seeds_used = [int(current_seed) for current_seed, *_ in seed_weights]
            cell = {
                "kind": "null",
                "cohort": dataset,
                "architecture": arch,
                "aggregation": aggregation,
                "seeds": seeds_used,
                "weight_types": list(weight_types),
                "n_subjects": int(n_subjects),
                "n_permutations": int(n_permutations),
            }
            resolved = {
                "cell": cell,
                "scaler": scaler,
                "channels": channels,
                "downsample_factor": int(downsample_factor),
                "null_rng_seed": int(null_rng_seed),
                "strict_complete": bool(strict_complete),
                "git": git_state(project_root),
                "environment": environment_snapshot(),
                "subject_ids": sorted(sampled["id"].astype(str).tolist()),
            }
            resolved_path = out_dir / "resolved_run.json"
            resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
            subject_path = out_dir / "subject_ids.json"
            subject_path.write_text(json.dumps(resolved["subject_ids"], indent=2) + "\n")
            artifact_paths = [
                path for path in sorted(out_dir.iterdir())
                if path.is_file() and path.name != "null_cell_manifest.json"
            ]
            required_npz = list(out_dir.glob("permutation_region_stats_*.npz"))
            if strict_complete and len(required_npz) != len(weight_types):
                raise RuntimeError(
                    f"Expected {len(weight_types)} permutation NPZ files, found {len(required_npz)}"
                )
            cell_manifest = {
                "schema_version": 1,
                "status": "complete",
                "cell": cell,
                "artifacts": [
                    {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in artifact_paths
                ],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            temporary = out_dir / "null_cell_manifest.json.tmp"
            temporary.write_text(json.dumps(cell_manifest, indent=2, sort_keys=True) + "\n")
            temporary.replace(out_dir / "null_cell_manifest.json")

    print(f"\nDone. Results written to {result_dir}/group_diff/voxel_importance/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map CNN feature importance to voxel regions via gradient attribution.\n\n"
            "Outputs (in <result_dir>/group_diff/voxel_importance/<agg>/):\n"
            "  region_importance_*.csv         per-region signed + absolute importance\n"
            "                                   (+ mean_null_*/mean_corrected_*/mean_z_*\n"
            "                                   columns if --n-permutations > 0)\n"
            "  region_barplot_*.png            top-30 regions, colored by direction\n"
            "  cnn_vs_roi_comparison_*.csv     CNN importance vs FS ROI Cohen's d\n"
            "  cnn_vs_roi_scatter_*.png        scatter of the above\n"
            "  voxel_map_{signed,abs}_*.nii.gz full 256³ maps (--save-voxel-map only)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["nki", "ppmi"], default="nki",
        help="Dataset: 'nki' (Child vs Adult) or 'ppmi' (PD vs HC). Default: nki.",
    )
    parser.add_argument(
        "--arch", default="double_conv", choices=["double_conv", "cov_pool"],
        help="CNN architecture. Default: double_conv.",
    )
    parser.add_argument(
        "--scaler", default="minmax",
        help="Scaler used during feature extraction. Default: minmax.",
    )
    parser.add_argument(
        "--channels", default="t1_rank_sobel",
        help="Channels used during feature extraction (underscore-joined). Default: t1_rank_sobel.",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=50,
        help="Number of subjects to sample for gradient averaging. Default: 50.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Maximum number of CNN seeds to use. Default: all available.",
    )
    parser.add_argument(
        "--agg", nargs="+",
        default=list(AGGREGATION_SPECS.keys()),
        help="Aggregation(s) to process. Default: all.",
    )
    parser.add_argument(
        "--save-voxel-map", action="store_true",
        help="Also save full 256³ voxel maps as NIfTI files.",
    )
    parser.add_argument(
        "--skip-lme", action="store_true",
        help="Skip aggregations that require LME fitting (lme_slope, lme_slope_change).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device for gradient computation. Default: cpu.",
    )
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="Spatial downsampling factor applied to the input before the forward "
             "pass (default 2: 256³→128³). Reduces first-layer activations from "
             "4.3 GB to 0.54 GB, making GPU feasible without autocast or "
             "gradient checkpointing. Gradient is upsampled back to 256³. "
             "Use 1 to disable.",
    )
    parser.add_argument(
        "--weight-types", nargs="+", default=["tstat"],
        choices=["tstat", "shap", "hybrid"],
        help="Importance-weight derivation(s) to attribute and compare. "
             "'tstat': Welch t-statistic (existing behavior). "
             "'shap': signed mean TreeSHAP value from a RandomForestClassifier. "
             "'hybrid': RF permutation-importance x sign(mean_a - mean_b). "
             "'shap'/'hybrid' share one RF fit per (seed, aggregation). Default: tstat.",
    )
    parser.add_argument(
        "--rf-n-estimators", type=int, default=200,
        help="Number of trees for the RandomForestClassifier backing 'shap'/'hybrid' "
             "weight types. Default: 200.",
    )
    parser.add_argument(
        "--rf-permutation-repeats", type=int, default=10,
        help="Number of shuffles per feature for sklearn permutation_importance, "
             "used by the 'hybrid' weight type. Default: 10.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf", type=int, default=5,
        help="min_samples_leaf for the RandomForestClassifier backing 'shap'/'hybrid'. "
             "Prevents train_acc=1.0 (which makes 'hybrid' permutation importance "
             "degenerate). Default: 5.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=0,
        help="Number of group-label permutations for the architecture-bias "
             "permutation null (Phase 4b, group_perm design). 0 disables "
             "null-correction (default). Each permutation re-derives ALL "
             "requested --weight-types from permuted group labels "
             "(refitting the RandomForestClassifier for 'shap'/'hybrid'), "
             "rank-normalizes, and backpropagates -- expensive: budget "
             "sbatch time/array-sizing accordingly (roughly "
             "n_permutations x n_seeds x [RF-refit + backprop] extra work).",
    )
    parser.add_argument(
        "--null-rng-seed", type=int, default=0,
        help="Base RNG seed for group-label permutations (offset by CNN seed "
             "per seed for reproducibility). Default: 0.",
    )
    parser.add_argument(
        "--strict-complete", action="store_true",
        help="Confirmatory mode: fail nonzero on any missing seed, subject, map, "
             "weight type, permutation, or CUDA error; disables CPU fallback.",
    )
    parser.add_argument(
        "--validation-output-dir", type=Path, default=None,
        help="Write confirmatory artifacts under this versioned directory instead "
             "of the exploratory Phase 4b output tree.",
    )
    args = parser.parse_args()

    run(
        project_root=ROOT,
        dataset=args.dataset,
        arch=args.arch,
        scaler=args.scaler,
        channels=args.channels,
        n_subjects=args.n_subjects,
        n_seeds=args.n_seeds,
        aggregations=args.agg,
        save_voxel_map=args.save_voxel_map,
        device=args.device,
        skip_lme=args.skip_lme,
        downsample_factor=args.downsample,
        weight_types=args.weight_types,
        rf_n_estimators=args.rf_n_estimators,
        rf_permutation_repeats=args.rf_permutation_repeats,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        n_permutations=args.n_permutations,
        null_rng_seed=args.null_rng_seed,
        strict_complete=args.strict_complete,
        validation_output_dir=args.validation_output_dir,
    )


if __name__ == "__main__":
    main()
