"""T2-M2: Feature-wise partial correlation with NP3TOT (UPDRS-III motor score).

For each (extractor, aggregation) cell, builds per-subject aggregated vectors
(PD subjects only) and computes the partial Spearman correlation of each feature
element with NP3TOT_first, controlling for age and sex.

Method:
  1. Residualise both features and NP3TOT against [age, sex] using OLS.
  2. Compute Spearman r between residualised features and residualised NP3TOT.
  3. Apply BH-FDR correction within each (aggregation, extractor) cell.

Outputs (written to <result_dir>/partial_corr/):
  summary.csv                     — one row per (extractor_key, aggregation):
                                    n_sig, frac_sig, median_r, max_abs_r
  features_fs_roi_<agg>.csv       — per-feature partial r, p_raw, p_fdr for FS ROI
  volcano_fs_roi_<agg>.png        — volcano plot (partial r × −log10 p_fdr)
  heatmap_frac_sig.png            — heatmap: frac_sig across aggregation × extractor_group

Usage:
    python3 -m brainage_agg.analysis.partial_corr                    # sequential NKI/PPMI
    python3 brainage_agg/analysis/partial_corr.py --dataset ppmi
    python3 brainage_agg/analysis/partial_corr.py --extractor-idx N  # SLURM array
    python3 brainage_agg/analysis/partial_corr.py --merge
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Residualisation helpers
# ---------------------------------------------------------------------------

def _residualise(Y: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Remove confound signal from Y via OLS. Y can be (n,) or (n, d)."""
    reg = LinearRegression().fit(confounds, Y)
    return Y - reg.predict(confounds)


def build_confound_matrix(manifest_sub: pd.DataFrame) -> np.ndarray:
    """Build confound matrix [age, sex_indicator] for each subject in manifest_sub.

    Returns (n_subjects, n_confounds).  sex is dummy-coded: 0=F, 1=M.
    """
    import json as _json
    ages = []
    for v in manifest_sub["ages"]:
        parsed = _json.loads(v) if isinstance(v, str) else v
        ages.append(float(np.mean(parsed)))
    sex_map = {"M": 1.0, "F": 0.0, "1": 1.0, "0": 0.0}
    sex_col = manifest_sub.get("sex", pd.Series(["F"] * len(manifest_sub))).values
    sexes   = [sex_map.get(str(s), 0.0) for s in sex_col]
    return np.column_stack([ages, sexes])


# ---------------------------------------------------------------------------
# Partial Spearman correlation (per feature)
# ---------------------------------------------------------------------------

def partial_spearman_with_updrs(
    X: np.ndarray,
    y: np.ndarray,
    confounds: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Partial Spearman r between each feature column and y, controlling for confounds.

    Args:
        X:          (n_subjects, d)  aggregated feature matrix
        y:          (n_subjects,)    NP3TOT scores
        confounds:  (n_subjects, k)  confound matrix (age, sex)
        feature_names: optional list of length d

    Returns DataFrame with columns:
        feature, partial_r, p_raw, p_fdr, significant
    """
    from statsmodels.stats.multitest import multipletests

    n, d = X.shape
    if feature_names is None:
        feature_names = [f"f_{j:04d}" for j in range(d)]

    # Rank-transform then residualise (= partial Spearman via partialisation of ranks)
    X_ranked = np.apply_along_axis(stats.rankdata, 0, X).astype(float)
    y_ranked = stats.rankdata(y).astype(float)

    C = StandardScaler().fit_transform(confounds)
    X_res = _residualise(X_ranked, C)
    y_res = _residualise(y_ranked, C)

    rs     = np.zeros(d)
    p_raw  = np.ones(d)
    for j in range(d):
        xj = X_res[:, j]
        if xj.std() < 1e-10:
            p_raw[j] = np.nan
            continue
        r, p = stats.pearsonr(xj, y_res)
        rs[j]    = float(r)
        p_raw[j] = float(p)

    p_fdr = np.full(d, np.nan)
    valid = ~np.isnan(p_raw)
    if valid.sum() > 0:
        _, p_fdr_v, _, _ = multipletests(p_raw[valid], method="fdr_bh")
        p_fdr[valid] = p_fdr_v

    return pd.DataFrame({
        "feature":     feature_names,
        "partial_r":   rs,
        "p_raw":       p_raw,
        "p_fdr":       p_fdr,
        "significant": p_fdr < 0.05,
    })


# ---------------------------------------------------------------------------
# Aggregation builder (reuses group_diff logic, PD only)
# ---------------------------------------------------------------------------

def build_aggregated_vectors_pd(
    manifest: pd.DataFrame,
    subject_data: dict,
    aggregation: str,
    lme_subjects: np.ndarray | None = None,
    lme_slopes: np.ndarray | None = None,
    target_col: str = "NP3TOT_first",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build aggregated X for PD subjects with valid NP3TOT.

    Returns (X, y_updrs, subject_ids, manifest_subset).
    """
    import json as _json
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.agg import aggregations as agg_mod
    from brainage_agg.agg.aggregations import AGGREGATION_SPECS

    spec = AGGREGATION_SPECS.get(aggregation, {"cohort": "all", "needs_ages": False})
    df   = manifest[manifest["band"] == "PD"].copy()
    cohort_col = "cohort_span1" if "cohort_span1" in df.columns else "cohort_all"

    if aggregation == "cross_sectional":
        if "cohort_span1" in df.columns:
            df = df[df["cohort_span1"]]
    elif spec.get("cohort") == "concat":
        df_c = df[df["cohort_concat"]]
        df = df_c if len(df_c) > 0 else df[df[cohort_col]]
    else:
        df = df[df[cohort_col]]

    if aggregation in ("annualized_rate", "difference", "lme_slope_change"):
        df = df[~df["exclude_arm_b"]]

    # Keep only subjects with a valid NP3TOT
    df = df.dropna(subset=[target_col])

    sid_manifest = df.set_index("subject_id")
    rows_X, rows_y, rows_sid = [], [], []

    for sid in df["subject_id"]:
        if sid not in subject_data:
            continue
        mrow  = sid_manifest.loc[sid]
        feats = subject_data[sid]["features"]
        ages  = subject_data[sid]["ages"]

        if aggregation == "cross_sectional":
            x = feats[0]
        elif aggregation == "mean":
            x = agg_mod.mean(feats)
        elif aggregation == "concatenation":
            t_mid_idx = int(mrow["t_mid_idx"])
            if t_mid_idx < 0:
                x = np.concatenate([feats[0], feats[-1]])
            else:
                row_indices = mrow["row_indices"]
                if isinstance(row_indices, str):
                    row_indices = _json.loads(row_indices)
                t_mid_local = list(row_indices).index(t_mid_idx)
                x = agg_mod.concatenation(feats, 0, t_mid_local, len(feats) - 1)
        elif aggregation == "annualized_rate":
            try:
                x = agg_mod.annualized_rate(feats, ages)
            except ValueError:
                continue
        elif aggregation in ("lme_slope", "lme_slope_change"):
            if lme_subjects is None:
                continue
            try:
                x = agg_mod.lme_slope(sid, lme_subjects, lme_slopes)
            except KeyError:
                continue
        elif aggregation == "difference":
            x = agg_mod.difference(feats)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        rows_X.append(x)
        rows_y.append(float(mrow[target_col]))
        rows_sid.append(sid)

    if not rows_X:
        return np.empty((0, 0)), np.array([]), np.array([]), df

    return (
        np.array(rows_X, dtype=np.float64),
        np.array(rows_y, dtype=np.float64),
        np.array(rows_sid),
        df.set_index("subject_id").loc[rows_sid].reset_index(),
    )


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------

def make_volcano(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    label_top_n: int = 10,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x   = df["partial_r"].values
    y   = -np.log10(df["p_fdr"].clip(lower=1e-300).values)
    sig = df["significant"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[~sig],             y[~sig],             s=8,  alpha=0.3, color="grey",      label="n.s.")
    ax.scatter(x[sig & (x > 0)],   y[sig & (x > 0)],   s=14, alpha=0.8, color="tomato",    label="r > 0  (higher feature → more severe)")
    ax.scatter(x[sig & (x <= 0)],  y[sig & (x <= 0)],  s=14, alpha=0.8, color="steelblue", label="r < 0  (lower feature → more severe)")

    if df["significant"].any():
        top_idx = df["significant"] & (
            df["partial_r"].abs() >= df.loc[df["significant"], "partial_r"].abs().nlargest(label_top_n).min()
        )
        for _, row in df[top_idx].iterrows():
            ax.annotate(
                row["feature"],
                xy=(row["partial_r"], -np.log10(max(row["p_fdr"], 1e-300))),
                fontsize=5, ha="center", va="bottom",
            )

    ax.axhline(-np.log10(0.05), color="black", linewidth=0.8, linestyle="--", label="FDR=0.05")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Partial Spearman r  (controlling for age + sex)")
    ax.set_ylabel("−log₁₀(p_fdr)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def make_summary_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    agg_order = ["cross_sectional", "mean", "concatenation", "annualized_rate",
                 "difference", "lme_slope_change"]

    pivot = (
        summary.groupby(["extractor_group", "aggregation"])["frac_sig"]
        .mean()
        .reset_index()
        .pivot(index="extractor_group", columns="aggregation", values="frac_sig")
    )
    cols  = [c for c in agg_order if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.4), max(4, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Significant fraction (FDR<0.05)"})
    ax.set_title("Partial Spearman r with NP3TOT — significant fraction\n(controlling for age + sex, PD only)")
    ax.set_ylabel("Extractor group")
    ax.set_xlabel("Aggregation")
    plt.tight_layout()
    fig.savefig(out_dir / "heatmap_frac_sig.png", dpi=150)
    plt.close(fig)
    print("    Saved heatmap_frac_sig.png")


# ---------------------------------------------------------------------------
# Per-extractor metadata helper
# ---------------------------------------------------------------------------

def _extractor_metadata(npz_path: Path, is_roi: bool) -> dict:
    parts = {
        seg.split("-", 1)[0]: seg.split("-", 1)[1]
        for seg in npz_path.stem.split("__")[1:]
        if "-" in seg
    }
    arch = parts.get("model", "double_conv")
    return {
        "extractor":       npz_path.stem,
        "extractor_group": "FS ROI" if is_roi
                           else f"{arch} / {parts.get('scaler','?')} / {parts.get('channels','?')}",
        "cnn_arch":        arch,
        "cnn_scaler":      parts.get("scaler", "none"),
        "cnn_channels":    parts.get("channels", "all_roi"),
        "cnn_seed":        int(parts.get("seed", "0")),
    }


# ---------------------------------------------------------------------------
# Core: process one extractor
# ---------------------------------------------------------------------------

PARTIAL_CORR_AGGS = [
    "cross_sectional",
    "mean",
    "concatenation",
    "annualized_rate",
    "difference",
    "lme_slope_change",
]


def process_one_extractor(
    npz_path: Path,
    manifest: pd.DataFrame,
    out_dir: Path,
    target_col: str = "NP3TOT_first",
    skip_lme_aggs: bool = False,
) -> list[dict]:
    """Process one extractor NPZ. Returns summary rows (one per aggregation)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.features.loader import load_features, align_to_manifest, is_fs_roi_file
    from brainage_agg.agg.lme import make_lme_estimator
    from brainage_agg.analysis.group_diff import load_feature_names, concat_feature_names, generic_names

    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    is_roi = is_fs_roi_file(npz_path)
    meta   = _extractor_metadata(npz_path, is_roi)
    extractor_id = meta["extractor"]

    X_npz, meta_df, _ = load_features(npz_path)
    subject_data       = align_to_manifest(X_npz, meta_df, manifest)
    n_feat             = X_npz.shape[1]
    base_names         = load_feature_names(npz_path) if is_roi else None

    # --- Flat arrays for LME fitting (PD only) ---
    cohort_col = "cohort_span1" if "cohort_span1" in manifest.columns else "cohort_all"
    pd_manifest = manifest[(manifest["band"] == "PD") & manifest[cohort_col]].copy()
    flat_X, flat_ages, flat_sids = [], [], []
    for _, mrow in pd_manifest.iterrows():
        sid = mrow["subject_id"]
        if sid not in subject_data:
            continue
        sd = subject_data[sid]
        for feat, age in zip(sd["features"], sd["ages"]):
            flat_X.append(feat); flat_ages.append(age); flat_sids.append(sid)

    flat_X    = np.array(flat_X,    dtype=np.float64)
    flat_ages = np.array(flat_ages, dtype=np.float64)
    flat_sids = np.array(flat_sids)

    # --- Fit LME ---
    ages_c = flat_ages - flat_ages.mean()
    lme_subjects = lme_slopes = None
    if not skip_lme_aggs and len(flat_X) > 0:
        lme_est = make_lme_estimator(n_feat, threshold=700)
        print(f"  Fitting LME ({type(lme_est).__name__})...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lme_est.fit(flat_X, ages_c, flat_sids)
        lme_subjects, lme_slopes = lme_est.transform(flat_X, ages_c, flat_sids)
        print("done")

    summary_rows = []
    aggs = [a for a in PARTIAL_CORR_AGGS
            if not (skip_lme_aggs and a in ("lme_slope", "lme_slope_change"))]

    for aggregation in aggs:
        X_agg, y_updrs, sids_agg, mdf_sub = build_aggregated_vectors_pd(
            manifest, subject_data, aggregation,
            lme_subjects=lme_subjects, lme_slopes=lme_slopes,
            target_col=target_col,
        )

        if len(X_agg) < 10:
            print(f"  SKIP {aggregation}: only {len(X_agg)} subjects with valid data")
            continue

        d_agg = X_agg.shape[1]
        if aggregation == "concatenation":
            base = base_names or generic_names(n_feat)
            if d_agg == 3 * n_feat:
                agg_feat_names = concat_feature_names(base_names, n_feat)
            elif d_agg == 2 * n_feat:
                agg_feat_names = [f"{n}_t_first" for n in base] + [f"{n}_t_last" for n in base]
            else:
                agg_feat_names = generic_names(d_agg)
        elif base_names:
            agg_feat_names = base_names
        else:
            agg_feat_names = generic_names(d_agg)

        # Build confound matrix for subjects in this aggregation
        confounds = build_confound_matrix(mdf_sub)

        pcorr_df = partial_spearman_with_updrs(X_agg, y_updrs, confounds, agg_feat_names)
        n_sig = int(pcorr_df["significant"].sum())
        print(f"  {aggregation:20s}  n={len(X_agg)}  n_sig={n_sig:4d}/{d_agg}  "
              f"median_r={pcorr_df['partial_r'].abs().median():.3f}")

        row = {
            **meta,
            "aggregation":   aggregation,
            "target_col":    target_col,
            "n_subjects":    len(X_agg),
            "n_features":    d_agg,
            "n_significant": n_sig,
            "frac_sig":      n_sig / d_agg,
            "median_abs_r":  float(pcorr_df["partial_r"].abs().median()),
            "max_abs_r":     float(pcorr_df["partial_r"].abs().max()),
        }
        summary_rows.append(row)

        if is_roi:
            pcorr_df.to_csv(out_dir / f"features_fs_roi_{aggregation}.csv", index=False)
            make_volcano(
                pcorr_df,
                title=f"FS ROI — {aggregation} — partial r with NP3TOT (PD, n={len(X_agg)})",
                out_path=out_dir / f"volcano_fs_roi_{aggregation}.png",
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            partial_dir / f"{extractor_id}_summary.csv", index=False)

    return summary_rows


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(result_dir: Path, out_dir: Path | None = None) -> None:
    out_dir     = out_dir or (result_dir / "partial_corr")
    partial_dir = out_dir / "partial"

    parts = sorted(partial_dir.glob("*_summary.csv"))
    if not parts:
        print("No partial summary files found.")
        return

    summary_df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"summary.csv: {len(summary_df)} rows from {len(parts)} extractors")

    make_summary_heatmap(summary_df, out_dir)

    print("\n=== Mean frac_sig per extractor_group × aggregation ===")
    tbl = (summary_df.groupby(["extractor_group", "aggregation"])["frac_sig"]
           .mean().unstack("aggregation"))
    print(tbl.to_string())
    print(f"\nAll outputs in {out_dir}/")


# ---------------------------------------------------------------------------
# Sequential run
# ---------------------------------------------------------------------------

def run(
    result_dir: Path,
    feature_dir: Path,
    target_col: str = "NP3TOT_first",
    skip_lme_aggs: bool = False,
    out_dir: Path | None = None,
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.features.loader import list_cnn_files, is_fs_roi_file

    out_dir = out_dir or (result_dir / "partial_corr")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load UPDRS manifest, filter to PD
    manifest_path = result_dir / "manifest_with_updrs.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run add_updrs.py first.")
    manifest = pd.read_csv(manifest_path)
    import json as _json
    for col in ("ages", "row_indices"):
        if col in manifest.columns:
            manifest[col] = manifest[col].apply(
                lambda v: _json.loads(v) if isinstance(v, str) else v
            )

    fs_roi_path = feature_dir / "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
    all_paths   = [fs_roi_path] + list_cnn_files(feature_dir)

    for i, npz_path in enumerate(all_paths):
        print(f"\n[{i+1}/{len(all_paths)}] {Path(npz_path).name}")
        process_one_extractor(
            Path(npz_path), manifest, out_dir,
            target_col=target_col,
            skip_lme_aggs=skip_lme_aggs,
        )

    merge(result_dir, out_dir=out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial Spearman correlation with NP3TOT (UPDRS-III), PD subjects only.\n\n"
        "Three modes:\n"
        "  (default)         Sequential: process all extractors then merge.\n"
        "  --extractor-idx N SLURM array: process extractor N, write partial files.\n"
        "  --merge           Merge all partial files from a completed array job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",  type=Path, default=None,
        help="Result directory (default: brainage_agg/ppmi_outputs).",
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=None,
        help="Feature directory (default: PPMI_data/features).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for partial_corr results. "
             "Defaults to <result-dir>/partial_corr_<target> "
             "(or partial_corr for NP3TOT_first for backwards compatibility).",
    )
    parser.add_argument("--target",        type=str, default="NP3TOT_first",
                        help="Target column in manifest_with_updrs.csv.")
    parser.add_argument("--skip-lme",      action="store_true")
    parser.add_argument("--extractor-idx", type=int, default=None)
    parser.add_argument("--merge",         action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parents[2]
    result_dir   = args.result_dir  or project_root / "brainage_agg" / "ppmi_outputs"
    feature_dir  = args.feature_dir or project_root / "PPMI_data" / "features"

    def _default_out_dir(result_dir: Path, target: str) -> Path:
        subdir = "partial_corr" if target == "NP3TOT_first" else f"partial_corr_{target}"
        return result_dir / subdir

    out_dir = args.out_dir or _default_out_dir(result_dir, args.target)

    if args.merge:
        merge(result_dir, out_dir=out_dir)
        return

    if args.extractor_idx is not None:
        import sys, json as _json
        sys.path.insert(0, str(project_root))
        from brainage_agg.features.loader import list_cnn_files

        manifest_path = result_dir / "manifest_with_updrs.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found.")
        manifest = pd.read_csv(manifest_path)
        for col in ("ages", "row_indices"):
            if col in manifest.columns:
                manifest[col] = manifest[col].apply(
                    lambda v: _json.loads(v) if isinstance(v, str) else v
                )

        fs_roi_path = feature_dir / "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
        all_paths   = [fs_roi_path] + list_cnn_files(feature_dir)

        idx = args.extractor_idx
        if idx >= len(all_paths):
            print(f"Index {idx} out of range (total: {len(all_paths)}). Nothing to do.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = Path(all_paths[idx])
        print(f"Extractor {idx}/{len(all_paths)-1}: {npz_path.name}")
        process_one_extractor(
            npz_path, manifest, out_dir,
            target_col=args.target,
            skip_lme_aggs=args.skip_lme,
        )
        return

    run(
        result_dir=result_dir,
        feature_dir=feature_dir,
        target_col=args.target,
        skip_lme_aggs=args.skip_lme,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
