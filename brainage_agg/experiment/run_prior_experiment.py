"""Architectural prior + task feature importance experiment.

Modes:
  prior    — unit weights (np.ones) on frozen untrained CNN; captures
             inductive bias from architecture alone.
  age      — importance weights from brain age regression (RF or Ridge).
  sex      — importance weights from sex classification (RF or Ridge).
  severity — importance weights from UPDRS NP3TOT regression (PPMI only).

Output directories use {task}_{method}/ naming (e.g. age_rf/, sex_ridge/).
The prior/ directory always uses unit weights regardless of --method.

For each (dataset, arch, task, method):
  1. Compute / load cached mean brain image (one representative input).
  2. For each CNN seed: build frozen CNN → compute importance weights →
     backprop through mean brain → save per-seed voxel maps.
  3. Aggregate: mean map, std map, pairwise seed-agreement (Spearman ρ).
  4. Project to FreeSurfer aparc+aseg atlas regions.
  5. Cross-mode Spearman comparison across all completed subdirs.

Usage:
    python -m brainage_agg.experiment.run_prior_experiment
    python -m brainage_agg.experiment.run_prior_experiment \\
        --dataset nki --arch double_conv --modes prior age sex \\
        --method both --n-seeds 3 --device cpu
    python -m brainage_agg.experiment.run_prior_experiment --dataset ppmi \\
        --modes prior age severity --method rf
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import (
    compute_attribution_map,
    load_freesurfer_lut,
    project_to_atlas,
    subject_parcellation_path,
)
from brainage_agg.features.loader import load_features
from src.io import load_cohort, load_mgz_float32, load_mgz_mask
from src.models import CNN3D_CovPool, CNN3D_DoubleConv, freeze_model, init_xavier
from src.preprocessing import build_channels
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Helpers mirrored from run_voxel_importance.py
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


def _find_npz_files(feature_dir: Path, arch: str, scaler: str, channels: str) -> list[Path]:
    if arch == "double_conv":
        pattern = f"features__scaler-{scaler}__channels-{channels}__seed-*.npz"
    else:
        pattern = f"features__model-{arch}__scaler-{scaler}__channels-{channels}__seed-*.npz"
    return sorted(feature_dir.glob(pattern))


# ---------------------------------------------------------------------------
# Mean brain image
# ---------------------------------------------------------------------------

def _mask_cache_path(mean_brain_cache: Path) -> Path:
    stem = mean_brain_cache.stem.replace("mean_brain", "mean_brain_mask", 1)
    return mean_brain_cache.with_name(stem + ".npy")


def _compute_group_mask(cohort_dedup: pd.DataFrame, shape: tuple) -> np.ndarray:
    """Majority-vote brain mask across subjects (fast: only loads brainmask.mgz)."""
    mask_count = np.zeros(shape, dtype=np.float32)
    n = 0
    for _, row in cohort_dedup.iterrows():
        mask_path = str(row.get("mask_path", "") or "")
        if not (mask_path and Path(mask_path).exists()):
            continue
        try:
            mask_count += load_mgz_mask(mask_path).astype(np.float32)
            n += 1
        except Exception:
            pass
    return (mask_count / max(n, 1)) >= 0.5


def compute_mean_brain(
    cohort: pd.DataFrame,
    scaler: str,
    channels: list[str],
    cache_path: Path,
) -> np.ndarray:
    """Average preprocessed brain images across subjects → (C, H, W, D) float32.

    One session per participant (earliest timepoint) to avoid subject bias.
    Result is cached at cache_path so subsequent runs skip the computation.
    Also saves a majority-vote group brain mask alongside the mean brain.
    """
    msk_path = _mask_cache_path(cache_path)

    cohort_dedup = (
        cohort.sort_values("timepoint")
        .groupby("participant_id", sort=False)
        .first()
        .reset_index()
    )

    if cache_path.exists():
        print(f"  Loading cached mean brain from {cache_path.name}")
        result = np.load(cache_path)
        if not msk_path.exists():
            print("  Computing group brain mask (majority vote across subjects)...")
            shape = result.shape[1:]  # (H, W, D) from (C, H, W, D)
            group_mask = _compute_group_mask(cohort_dedup, shape)
            np.save(msk_path, group_mask)
            print(f"  Group mask saved to {msk_path.name}")
        return result

    print(f"  Computing mean brain from {len(cohort_dedup)} participants...")

    mean_arr: np.ndarray | None = None
    mask_count: np.ndarray | None = None
    n_loaded = 0
    for _, row in cohort_dedup.iterrows():
        try:
            img = load_mgz_float32(str(row["image_path"]))
            mask_path = str(row.get("mask_path", "") or "")
            mask = load_mgz_mask(mask_path) if mask_path and Path(mask_path).exists() else img > 0
            ch = build_channels(img, mask, scaler=scaler, channels=channels)  # (C, H, W, D)
            mean_arr = ch.astype(np.float64) if mean_arr is None else mean_arr + ch
            mask_count = mask.astype(np.float32) if mask_count is None else mask_count + mask
            n_loaded += 1
        except Exception as exc:
            warnings.warn(f"Skipping subject {row.get('participant_id', '?')}: {exc}")

    if mean_arr is None or n_loaded == 0:
        raise RuntimeError("No subjects could be loaded for mean brain computation.")

    group_mask = (mask_count / n_loaded) >= 0.5
    result = (mean_arr / n_loaded).astype(np.float32)
    result[:, ~group_mask] = 0.0  # zero out skull voxels present in <50% of subjects

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    np.save(msk_path, group_mask)
    print(f"  Mean brain computed from {n_loaded} participants and cached.")
    return result


# ---------------------------------------------------------------------------
# Importance weights: RF and Ridge
# ---------------------------------------------------------------------------

def compute_rf_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "regression",
    n_estimators: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    """Fit RF on full data, return feature_importances_ (sums to 1)."""
    if task_type == "classification":
        est = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    else:
        est = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    est.fit(X, y)
    return est.feature_importances_.astype(np.float32)


def compute_ridge_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "regression",
    alphas: list[float] | None = None,
) -> np.ndarray:
    """Fit Ridge or LogisticRegression with 5-fold CV regularisation selection, return |coef|.

    Regression  → Ridge(alpha); GridSearch over alpha in [0.01 … 1000].
    Classification → LogisticRegression(C=1/alpha, l2); GridSearch over C in [0.001 … 100].
    """
    if task_type == "classification":
        if alphas is None:
            Cs = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
        else:
            Cs = [1.0 / a for a in alphas]
        est = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
        gs = GridSearchCV(est, {"C": Cs}, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        gs.fit(X, y)
        best_C = gs.best_params_["C"]
        print(f"      LogisticReg best C={best_C} (cv balanced_accuracy={gs.best_score_:.3f})")
        coef = gs.best_estimator_.coef_
    else:
        if alphas is None:
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        est = Ridge()
        gs = GridSearchCV(est, {"alpha": alphas}, cv=5, scoring="r2", n_jobs=-1)
        gs.fit(X, y)
        best_alpha = gs.best_params_["alpha"]
        print(f"      Ridge best alpha={best_alpha} (cv r2={gs.best_score_:.3f})")
        coef = gs.best_estimator_.coef_

    if coef.ndim > 1:
        coef = coef[0]  # binary LogisticRegression returns shape (1, n_features)
    return np.abs(coef).astype(np.float32)


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------

def _align_labels(
    task: str,
    dataset: str,
    X: np.ndarray,
    meta_df: pd.DataFrame,
    updrs_csv: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return (X_clean, y_clean, task_type) with NaN/invalid rows removed.

    Returns (None, None, "") if task is not applicable for this dataset.
    """
    if task == "age":
        y = meta_df["age"].values.astype(np.float32)
        valid = ~np.isnan(y)
        return X[valid], y[valid], "regression"

    if task == "sex":
        sex_raw = meta_df["sex"].values
        y = np.full(len(sex_raw), np.nan, dtype=np.float32)
        for i, s in enumerate(sex_raw):
            if isinstance(s, str):
                sl = s.strip().upper()
                if sl in ("M", "MALE"):
                    y[i] = 0.0
                elif sl in ("F", "FEMALE"):
                    y[i] = 1.0
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            warnings.warn(f"Only {valid.sum()} subjects with valid sex labels — skipping.")
            return None, None, ""
        return X[valid], y[valid].astype(int), "classification"

    if task == "severity":
        if dataset != "ppmi" or updrs_csv is None or not updrs_csv.exists():
            return None, None, ""
        updrs = pd.read_csv(updrs_csv)
        merged = meta_df.reset_index(drop=True)[["id"]].merge(
            updrs[["subject_id", "NP3TOT_first"]].drop_duplicates("subject_id"),
            left_on="id",
            right_on="subject_id",
            how="left",
        )
        y = merged["NP3TOT_first"].values.astype(np.float32)
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            warnings.warn(
                f"Only {valid.sum()} subjects with NP3TOT_first labels — skipping severity."
            )
            return None, None, ""
        return X[valid], y[valid], "regression"

    return None, None, ""  # prior mode — caller uses unit weights


# ---------------------------------------------------------------------------
# Reference parcellation
# ---------------------------------------------------------------------------

def _load_reference_parcellation(
    cohort: pd.DataFrame,
    fs_root: str | None,
) -> np.ndarray | None:
    if fs_root is None:
        return None
    for _, row in cohort.iterrows():
        parc_path = subject_parcellation_path(fs_root, row["directory"])
        if parc_path.exists():
            vol = np.asarray(nib.load(str(parc_path)).dataobj, dtype=np.int32)
            return vol[..., 0] if vol.ndim > 3 else vol
    return None


# ---------------------------------------------------------------------------
# Seed agreement
# ---------------------------------------------------------------------------

def compute_seed_agreement(seed_maps: list[np.ndarray]) -> pd.DataFrame:
    """Pairwise Spearman ρ between flattened abs maps → n_seeds × n_seeds DataFrame."""
    n = len(seed_maps)
    flat = [m.ravel() for m in seed_maps]
    corr = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(flat[i], flat[j])
            corr[i, j] = corr[j, i] = float(rho)
    return pd.DataFrame(corr)


# ---------------------------------------------------------------------------
# Single-mode + method experiment loop
# ---------------------------------------------------------------------------

def run_single_mode(
    task: str,
    method: str,
    arch: str,
    dataset: str,
    npz_files: list[Path],
    channels: list[str],
    mean_brain_arr: np.ndarray,
    atlas: np.ndarray | None,
    lut: dict[int, str],
    output_dir: Path,
    device: str,
    updrs_csv: Path | None,
    top_k_pct: float | None = None,
) -> None:
    """Run attribution for one (task, method) pair across all available seeds.

    Output directory: {output_dir}/{task}_{method}/ for task modes,
    {output_dir}/prior/ for the prior mode.
    """
    if task == "prior":
        dir_key = "prior"
    elif top_k_pct is not None:
        dir_key = f"{task}_{method}_top{top_k_pct:g}pct"
    else:
        dir_key = f"{task}_{method}"

    mode_dir = output_dir / dir_key
    mode_dir.mkdir(parents=True, exist_ok=True)

    mean_brain_tensor = torch.from_numpy(mean_brain_arr[np.newaxis]).float()

    seed_signed_maps: list[np.ndarray] = []
    seed_abs_maps: list[np.ndarray] = []
    region_dfs: list[pd.DataFrame] = []

    for npz_path in npz_files:
        X_npz, meta_df, meta_dict = load_features(npz_path)
        seed = int(meta_dict.get("seed", 0))
        n_features = X_npz.shape[1]
        print(f"    seed={seed}  n_features={n_features}")

        # --- Importance weights ---
        if task == "prior":
            importance_weights = np.ones(n_features, dtype=np.float32)
        else:
            X_clean, y_clean, task_type = _align_labels(task, dataset, X_npz, meta_df, updrs_csv)
            if X_clean is None:
                print(f"      No labels for task={task!r}, dataset={dataset!r} — skipping.")
                return

            n_samples = len(X_clean)
            label_info = (
                f"{y_clean.mean():.2f} ± {y_clean.std():.2f}"
                if task_type == "regression"
                else f"classes={np.unique(y_clean).tolist()}"
            )
            print(f"      Fitting {method} on {n_samples} samples ({label_info})...")

            if method == "rf":
                importance_weights = compute_rf_importance(X_clean, y_clean, task_type)
            elif method == "ridge":
                importance_weights = compute_ridge_importance(X_clean, y_clean, task_type)
            else:
                raise ValueError(f"Unknown method: {method!r}")

            if top_k_pct is not None:
                threshold = np.percentile(importance_weights, 100.0 - top_k_pct)
                n_kept = int((importance_weights >= threshold).sum())
                importance_weights = np.where(
                    importance_weights >= threshold, importance_weights, 0.0
                ).astype(np.float32)
                print(f"      top-{top_k_pct:g}%: keeping {n_kept}/{len(importance_weights)} features")

            np.save(mode_dir / f"seed-{seed}_{method}_importances.npy", importance_weights)

        # --- Build frozen CNN for this seed ---
        model = _build_cnn(arch, channels, seed, device)

        # --- Gradient attribution ---
        signed_map, abs_map = compute_attribution_map(
            model, mean_brain_tensor, importance_weights, device
        )

        # --- Per-seed outputs ---
        np.save(mode_dir / f"seed-{seed}_signed_map.npy", signed_map)
        np.save(mode_dir / f"seed-{seed}_abs_map.npy", abs_map)

        if atlas is not None:
            res_signed, res_abs = project_to_atlas(signed_map, abs_map, atlas)
            rows = [
                {
                    "label_id": lbl,
                    "region": lut.get(lbl, f"label_{lbl}"),
                    "n_voxels": res_signed[lbl][1],
                    "mean_signed": res_signed[lbl][0],
                    "mean_abs": res_abs[lbl][0],
                }
                for lbl in res_signed
            ]
            region_df = pd.DataFrame(rows).sort_values("mean_abs", ascending=False)
            region_df.to_csv(mode_dir / f"seed-{seed}_atlas_regions.csv", index=False)
            region_dfs.append(region_df.set_index("label_id"))

        seed_signed_maps.append(signed_map)
        seed_abs_maps.append(abs_map)

        del model
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    if not seed_signed_maps:
        return

    # --- Aggregate ---
    stack_signed = np.stack(seed_signed_maps, axis=0)
    stack_abs = np.stack(seed_abs_maps, axis=0)
    np.save(mode_dir / "mean_signed_map.npy", stack_signed.mean(axis=0).astype(np.float32))
    np.save(mode_dir / "mean_abs_map.npy", stack_abs.mean(axis=0).astype(np.float32))
    np.save(mode_dir / "std_abs_map.npy", stack_abs.std(axis=0).astype(np.float32))

    # --- Seed agreement ---
    if len(seed_abs_maps) > 1:
        agreement_df = compute_seed_agreement(seed_abs_maps)
        agreement_df.to_csv(mode_dir / "seed_agreement.csv", index=False)
        off_diag = agreement_df.values[np.triu_indices(len(agreement_df), k=1)]
        print(f"    Seed agreement — mean ρ: {off_diag.mean():.3f}  (min {off_diag.min():.3f})")

    # --- Mean region summary ---
    if region_dfs:
        all_labels = sorted(set.intersection(*[set(df.index) for df in region_dfs]))
        mean_signed = np.mean(
            [df.reindex(all_labels)["mean_signed"].values for df in region_dfs], axis=0
        )
        mean_abs = np.mean(
            [df.reindex(all_labels)["mean_abs"].values for df in region_dfs], axis=0
        )
        std_abs = np.std(
            [df.reindex(all_labels)["mean_abs"].values for df in region_dfs], axis=0
        )
        ref = region_dfs[0].reindex(all_labels)
        summary = pd.DataFrame({
            "label_id": all_labels,
            "region": ref["region"].values,
            "n_voxels": ref["n_voxels"].values,
            "mean_signed": mean_signed,
            "mean_abs": mean_abs,
            "std_abs": std_abs,
        }).sort_values("mean_abs", ascending=False)
        summary.to_csv(mode_dir / "mean_atlas_regions.csv", index=False)

    print(f"    {dir_key!r} complete — {mode_dir}")


# ---------------------------------------------------------------------------
# Cross-experiment comparison
# ---------------------------------------------------------------------------

def compute_cross_comparison(output_dir: Path, dir_keys: list[str]) -> None:
    """Pairwise Spearman ρ between mean abs maps (voxel + region level)."""
    voxel_maps: dict[str, np.ndarray] = {}
    region_maps: dict[str, pd.Series] = {}

    for key in dir_keys:
        vp = output_dir / key / "mean_abs_map.npy"
        if vp.exists():
            voxel_maps[key] = np.load(vp).ravel()
        rp = output_dir / key / "mean_atlas_regions.csv"
        if rp.exists():
            region_maps[key] = pd.read_csv(rp).set_index("label_id")["mean_abs"]

    if len(voxel_maps) < 2:
        return

    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    key_list = list(voxel_maps)
    voxel_rows, region_rows = [], []
    for i, k1 in enumerate(key_list):
        for k2 in key_list[i + 1:]:
            rho, pval = spearmanr(voxel_maps[k1], voxel_maps[k2])
            voxel_rows.append({"mode_a": k1, "mode_b": k2, "spearman_rho": round(rho, 4), "p_value": pval})
            if k1 in region_maps and k2 in region_maps:
                common = region_maps[k1].index.intersection(region_maps[k2].index)
                if len(common) >= 5:
                    rho_r, pval_r = spearmanr(region_maps[k1][common], region_maps[k2][common])
                    region_rows.append({
                        "mode_a": k1, "mode_b": k2,
                        "spearman_rho": round(rho_r, 4), "p_value": pval_r,
                        "n_regions": len(common),
                    })

    pd.DataFrame(voxel_rows).to_csv(comp_dir / "voxel_correlations.csv", index=False)
    if region_rows:
        pd.DataFrame(region_rows).to_csv(comp_dir / "region_correlations.csv", index=False)
    print(f"  Cross-experiment comparisons saved to {comp_dir}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    datasets: list[str],
    archs: list[str],
    tasks: list[str],
    methods: list[str],
    scaler: str,
    channels_str: str,
    n_seeds: int | None,
    output_base: Path,
    device: str,
    top_k_pct: float | None = None,
) -> None:
    channels = channels_str.split("_")

    for dataset in datasets:
        cfg_path = ROOT / "brainage_agg" / (
            "ppmi_config.yaml" if dataset == "ppmi" else "config.yaml"
        )
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)

        cohort_csv = ROOT / cfg["data"]["cohort_csv"]
        fs_root_cfg = cfg["data"].get("fs_root")
        fs_root = str(ROOT / fs_root_cfg) if fs_root_cfg else None
        feature_dir = Path(cfg["data"]["feature_dir"])
        if not feature_dir.is_absolute():
            feature_dir = ROOT / feature_dir

        cohort = (
            load_cohort(str(cohort_csv), freesurfer_root=fs_root)
            if fs_root
            else load_cohort(str(cohort_csv), use_csv_paths=True)
        )
        if cohort.empty:
            print(f"Warning: no valid subjects for dataset={dataset}, skipping.")
            continue

        updrs_csv = (
            ROOT / "brainage_agg" / "ppmi_outputs" / "manifest_with_updrs.csv"
            if dataset == "ppmi" else None
        )
        lut = load_freesurfer_lut(ROOT / "FS-LUT.txt")

        for arch in archs:
            print(f"\n=== dataset={dataset}  arch={arch} ===")
            output_dir = output_base / dataset / arch
            output_dir.mkdir(parents=True, exist_ok=True)

            cache_path = output_dir / f"mean_brain_{scaler}_{channels_str}.npy"
            mean_brain_arr = compute_mean_brain(cohort, scaler, channels, cache_path)

            atlas = _load_reference_parcellation(cohort, fs_root)
            if atlas is None:
                print("  Warning: no parcellation found — region projection will be skipped.")

            npz_files = _find_npz_files(feature_dir, arch, scaler, channels_str)
            if not npz_files:
                print(f"  No NPZ files found for arch={arch!r}, skipping.")
                continue
            if n_seeds is not None:
                npz_files = npz_files[:n_seeds]
            print(f"  Found {len(npz_files)} seed(s).")

            completed_dir_keys: list[str] = []

            for task in tasks:
                if task == "severity" and dataset != "ppmi":
                    print(f"  Skipping severity for dataset={dataset!r} (PPMI only).")
                    continue

                if task == "prior":
                    print(f"\n  -- task=prior --")
                    run_single_mode(
                        task="prior", method="unit",
                        arch=arch, dataset=dataset,
                        npz_files=npz_files, channels=channels,
                        mean_brain_arr=mean_brain_arr,
                        atlas=atlas, lut=lut,
                        output_dir=output_dir, device=device,
                        updrs_csv=updrs_csv,
                        top_k_pct=top_k_pct,
                    )
                    completed_dir_keys.append("prior")
                else:
                    for method in methods:
                        print(f"\n  -- task={task} method={method} --")
                        run_single_mode(
                            task=task, method=method,
                            arch=arch, dataset=dataset,
                            npz_files=npz_files, channels=channels,
                            mean_brain_arr=mean_brain_arr,
                            atlas=atlas, lut=lut,
                            output_dir=output_dir, device=device,
                            updrs_csv=updrs_csv,
                            top_k_pct=top_k_pct,
                        )
                        if top_k_pct is not None:
                            completed_dir_keys.append(f"{task}_{method}_top{top_k_pct:g}pct")
                        else:
                            completed_dir_keys.append(f"{task}_{method}")

            compute_cross_comparison(output_dir, completed_dir_keys)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Architectural prior + task feature importance experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", nargs="+", default=["nki", "ppmi"],
                        choices=["nki", "ppmi"], dest="datasets",
                        metavar="DATASET")
    parser.add_argument("--arch", nargs="+", default=["double_conv", "cov_pool"],
                        choices=["double_conv", "cov_pool"], dest="archs",
                        metavar="ARCH")
    parser.add_argument("--modes", nargs="+", default=["prior", "age", "sex", "severity"],
                        choices=["prior", "age", "sex", "severity"],
                        dest="tasks", metavar="TASK",
                        help="Tasks to run. 'prior' uses unit weights. "
                             "'severity' is PPMI-only.")
    parser.add_argument("--method", default="rf", choices=["rf", "ridge", "both"],
                        help="Importance method: rf (RandomForest), ridge (Ridge with CV alpha), "
                             "or both. Ignored for 'prior' task.")
    parser.add_argument("--scaler", default="minmax", choices=["minmax", "zscore", "robust"])
    parser.add_argument("--channels", default="t1",
                        help="Channel string matching NPZ naming, e.g. 't1' or 't1_rank_sobel'.")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Limit number of seeds processed (default: all found).")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "prior_experiment",
                        metavar="DIR")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=float, default=None, dest="top_k_pct",
                        metavar="PCT",
                        help="Keep only the top-PCT%% of importance features "
                             "(rest zeroed). E.g. --top-k 1 keeps the top 1%%.")
    args = parser.parse_args()

    methods = ["rf", "ridge"] if args.method == "both" else [args.method]

    run(
        datasets=args.datasets,
        archs=args.archs,
        tasks=args.tasks,
        methods=methods,
        scaler=args.scaler,
        channels_str=args.channels,
        n_seeds=args.n_seeds,
        output_base=args.output_dir,
        device=args.device,
        top_k_pct=args.top_k_pct,
    )


if __name__ == "__main__":
    main()
