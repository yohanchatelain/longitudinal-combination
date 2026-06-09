"""Map CNN feature importance to brain regions via gradient attribution.

Computes per-region importance comparable to FS ROI group-diff results
(features_fs_roi_<agg>.csv). For each aggregation type and each CNN seed:

  1. Build per-subject aggregated CNN feature vectors.
  2. Run Welch t-test (Child vs Adult / PD vs HC) → feature-level t-statistics w.
  3. Back-propagate w through the frozen CNN for each sampled subject to get
     voxel-level attribution maps (signed and absolute).
  4. Project maps to brain regions via aparc+aseg atlas from FreeSurfer.
  5. Average across subjects and seeds; save CSVs and plots.

Usage:
    python -m brainage_agg.experiment.run_voxel_importance
    python -m brainage_agg.experiment.run_voxel_importance \\
        --dataset nki --arch double_conv --scaler minmax --channels t1_rank_sobel \\
        --n-subjects 50 --n-seeds 5 --agg mean annualized_rate --save-voxel-map
    python -m brainage_agg.experiment.run_voxel_importance --dataset ppmi --skip-lme
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import (
    compute_attribution_map,
    load_freesurfer_lut,
    project_to_atlas,
    subject_parcellation_path,
)
from brainage_agg.analysis.group_diff import (
    _get_dataset_config,
    build_aggregated_vectors,
    run_option2_ttest,
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
# Importance weights (t-statistics) from group-difference analysis
# ---------------------------------------------------------------------------

def _compute_importance_weights(
    npz_path: Path,
    manifest: pd.DataFrame,
    aggregation: str,
    groups: tuple[str, str],
) -> np.ndarray | None:
    """Return per-CNN-feature t-statistics for one extractor × aggregation.

    Returns None if there is insufficient data for this aggregation.
    """
    spec = AGGREGATION_SPECS[aggregation]
    X_npz, meta_df, _ = load_features(npz_path)
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

    ttest_df = run_option2_ttest(X_agg, bands_agg, feature_names=None, groups=groups)
    return ttest_df["t_stat"].values.astype(np.float32)


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

def _region_importance_df(
    region_signed_accum: dict[int, list[float]],
    region_abs_accum: dict[int, list[float]],
    region_nvox: dict[int, int],
    lut: dict[int, str],
    groups: tuple[str, str],
) -> pd.DataFrame:
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

        rows.append({
            "label_id": label_id,
            "region_name": region_name,
            "hemisphere": hemisphere,
            "mean_abs_importance": mean_abs,
            "mean_signed_importance": mean_signed,
            "direction": f"{group_a} > {group_b}" if mean_signed > 0 else f"{group_b} > {group_a}",
            "n_voxels": n_vox,
        })
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
) -> None:
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
        print("No valid subjects after preprocessing — aborting.")
        return

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
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Compute per-seed importance weights ---
        seed_weights: list[tuple[Path, np.ndarray]] = []
        for npz_path in npz_files:
            print(f"  Importance weights: {npz_path.name} ...", end=" ", flush=True)
            w = _compute_importance_weights(npz_path, manifest, aggregation, groups)
            if w is None:
                print("skipped (insufficient data)")
                continue
            seed_weights.append((npz_path, w))
            print(f"done  n_feat={len(w)}")

        if not seed_weights:
            print(f"  No valid importance weights for '{aggregation}' — skipping.")
            continue

        # --- Accumulate attribution maps across seeds and subjects ---
        # Preprocessing is already done (cached_tensors); only gradient computation here.
        sum_signed_by_seed: list[np.ndarray] = []
        sum_abs_by_seed: list[np.ndarray] = []
        region_abs_accum:    dict[int, list[float]] = {}
        region_signed_accum: dict[int, list[float]] = {}
        region_nvox:         dict[int, int] = {}
        n_maps_total = 0

        for npz_path, w in seed_weights:
            _, _, meta = load_features(npz_path)
            seed = meta["seed"]
            print(
                f"  Seed {seed}: computing {len(cached_tensors)} attribution maps ...",
                flush=True,
            )
            t_seed = time.time()

            model = _build_cnn(arch, channels_list, seed, device)

            per_seed_signed: list[np.ndarray] = []
            per_seed_abs:    list[np.ndarray] = []

            for x_tensor, parc, _, brain_mask in cached_tensors:
                try:
                    signed_map, abs_map = compute_attribution_map(
                        model, x_tensor, w, device=device,
                        downsample_factor=downsample_factor,
                    )
                except RuntimeError as exc:
                    print(f"    Gradient computation failed: {exc}")
                    continue

                signed_map[~brain_mask] = 0.0
                abs_map[~brain_mask] = 0.0

                per_seed_signed.append(signed_map)
                per_seed_abs.append(abs_map)

                reg_signed, reg_abs = project_to_atlas(signed_map, abs_map, parc)
                for lid, (val, n_vox) in reg_abs.items():
                    region_abs_accum.setdefault(lid, []).append(val)
                    region_nvox[lid] = n_vox
                for lid, (val, _) in reg_signed.items():
                    region_signed_accum.setdefault(lid, []).append(val)

                n_maps_total += 1

            del model  # free GPU/CPU memory before next seed

            if per_seed_signed:
                sum_signed_by_seed.append(np.mean(per_seed_signed, axis=0))
                sum_abs_by_seed.append(np.mean(per_seed_abs, axis=0))
                print(
                    f"    {len(per_seed_signed)} maps in {time.time() - t_seed:.0f}s "
                    f"({(time.time() - t_seed) / max(1, len(per_seed_signed)):.1f}s/subject)"
                )

        if n_maps_total == 0:
            print(f"  No attribution maps produced for '{aggregation}' — skipping.")
            continue

        n_valid_seeds = len(sum_signed_by_seed)
        mean_signed_global = np.mean(sum_signed_by_seed, axis=0).astype(np.float32)
        mean_abs_global    = np.mean(sum_abs_by_seed,    axis=0).astype(np.float32)
        print(
            f"  Aggregated {n_maps_total} maps from {n_valid_seeds} seed(s); "
            f"global max |attribution| = {mean_abs_global.max():.4g}"
        )

        extractor_key = f"{arch}__{scaler}__{channels}__nsub{n_subjects}__{n_valid_seeds}seeds"

        # --- Save voxel maps as NIfTI ---
        if save_voxel_map and first_affine is not None:
            for tag, data in [("signed", mean_signed_global), ("abs", mean_abs_global)]:
                nii_path = out_dir / f"voxel_map_{tag}_{extractor_key}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine=first_affine), str(nii_path))
                print(f"  Saved {nii_path.name}")
        elif save_voxel_map:
            print("  Warning: could not save NIfTI (no subject affine captured)")

        # --- Region importance DataFrame ---
        region_df = _region_importance_df(
            region_signed_accum, region_abs_accum, region_nvox, lut, groups
        )
        csv_path = out_dir / f"region_importance_{extractor_key}.csv"
        region_df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path.name} ({len(region_df)} regions)")

        # --- Bar plot ---
        _make_barplot(
            region_df,
            title=(
                f"CNN voxel importance — {aggregation}\n"
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
                    title=f"CNN vs FS ROI — {aggregation}",
                    out_path=out_dir / f"cnn_vs_roi_scatter_{extractor_key}.png",
                )

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
    )


if __name__ == "__main__":
    main()
