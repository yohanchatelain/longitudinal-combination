"""Group difference analysis — extractor × aggregation comparison.

Supports two datasets:
  nki  — Child vs Adult brain age progression (default)
  ppmi — PD vs HC group difference in Parkinson's disease cohort

For each (extractor, aggregation) cell, builds per-subject aggregated vectors
(using the same aggregation functions as the prediction pipeline) and runs two
group-difference tests between the two groups:

  Option 2 — Feature-wise Welch t-test + BH-FDR on aggregated vector elements.
             "Which elements of this aggregation separate the groups?"

  Option 4 — Permutation MANOVA (Hotelling's T² in PCA space).
             "Does this aggregation globally separate the groups?"

Option 3 is a **raw-feature baseline** run once per extractor (not per aggregation):
  Option 3 — Per-feature Welch t-test on per-subject OLS slopes (pre-aggregation).
             Answers "how many raw features already show a slope difference?"
             For FS ROI: uses StatsmodelsLME for proper BLUP slopes.
             For CNN: uses VectorizedOLS (fast fallback).

CNN extractors (240 variants) are grouped by (arch, scaler, channels) and results
are averaged across their 10 seeds for reporting.

Outputs (written to <result_dir>/group_diff/):
  summary.csv                     — one row per (extractor_key, aggregation)
  option3_raw_baseline.csv        — per-feature raw slope test, one row per feature per extractor
  features_fs_roi_<agg>.csv       — per-element Option 2 results for FS ROI (with ROI names)
  heatmap_n_sig.png               — n_significant / n_total × (aggregation × extractor_group)
  heatmap_manova_p.png            — −log10(MANOVA p) × (aggregation × extractor_group)
  volcano_fs_roi_<agg>.png        — volcano for FS ROI per aggregation (Option 2)
  volcano_fs_roi_raw_baseline.png — volcano for FS ROI Option 3 raw slopes

Usage:
    python3 -m brainage_agg.analysis.group_diff
    python3 brainage_agg/analysis/group_diff.py --n-permutations 1000 --n-jobs 4
    python3 brainage_agg/analysis/group_diff.py --dataset ppmi
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------

def load_feature_names(npz_path: Path) -> list[str] | None:
    """Load feature names from JSON sidecar if it exists."""
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return data.get("feature_names")
    return None


def concat_feature_names(base_names: list[str] | None, n_feat: int) -> list[str]:
    """Return feature names for the concatenation aggregation [t_first, t_mid, t_last]."""
    if base_names is None:
        base_names = [f"f_{j:04d}" for j in range(n_feat)]
    return (
        [f"{n}_t_first" for n in base_names]
        + [f"{n}_t_mid"  for n in base_names]
        + [f"{n}_t_last" for n in base_names]
    )


def generic_names(n: int) -> list[str]:
    return [f"f_{j:04d}" for j in range(n)]


# ---------------------------------------------------------------------------
# Per-subject aggregated vector builder
# ---------------------------------------------------------------------------

def build_aggregated_vectors(
    manifest: pd.DataFrame,
    subject_data: dict,
    aggregation: str,
    lme_subjects: np.ndarray | None = None,
    lme_slopes: np.ndarray | None = None,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one aggregated vector per subject, return (X, bands, subject_ids).

    Uses the same logic as cv.py _build_agg_matrix but without CV:
    all eligible subjects, no train/test split.

    Returns:
        X:          (n_subjects, d)  aggregated feature matrix
        bands:      (n_subjects,)   group labels (e.g. 'Child'/'Adult' or 'PD'/'HC')
        sids:       (n_subjects,)   subject_ids
    """
    import sys, json as _json
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.agg import aggregations as agg_mod
    from brainage_agg.agg.aggregations import AGGREGATION_SPECS

    spec = AGGREGATION_SPECS[aggregation]
    df = manifest.copy()

    cohort_col = "cohort_span1" if "cohort_span1" in df.columns else "cohort_all"
    # Cohort filter: concatenation prefers ≥3 sessions but falls back to ≥2
    # when no subjects have ≥3 sessions (e.g. PPMI with 2 sessions each).
    if spec["cohort"] == "concat":
        df_concat = df[df["cohort_concat"]]
        df = df_concat if len(df_concat) > 0 else df[df[cohort_col]]
    else:
        df = df[df[cohort_col]]

    # Exclude zero-interval subjects for rate/diff aggregations
    if aggregation in ("annualized_rate", "difference", "lme_slope_change"):
        df = df[~df["exclude_arm_b"]]

    # Band filter: only subjects with known band
    df = df[df["band"].isin(list(groups))]

    if len(df) == 0:
        return np.empty((0, 0)), np.array([]), np.array([])

    sid_manifest = df.set_index("subject_id")
    rows_X, rows_band, rows_sid = [], [], []

    for sid in df["subject_id"]:
        sd = subject_data.get(sid)
        if sd is None:
            continue
        feats = sd["features"]   # (n_tp, n_feat)
        ages  = sd["ages"]
        mrow  = sid_manifest.loc[sid]

        if aggregation == "mean":
            x = agg_mod.mean(feats)

        elif aggregation == "concatenation":
            t_mid_global = int(mrow["t_mid_idx"])
            if t_mid_global < 0:
                # 2-session fallback: concatenate first and last only
                x = np.concatenate([feats[0], feats[-1]])
            else:
                global_rows = np.asarray(
                    mrow["row_indices"] if isinstance(mrow["row_indices"], list)
                    else _json.loads(mrow["row_indices"])
                )
                t_mid_local_arr = np.where(global_rows == t_mid_global)[0]
                if len(t_mid_local_arr) == 0:
                    continue
                x = agg_mod.concatenation(feats, 0, int(t_mid_local_arr[0]), len(feats) - 1)

        elif aggregation == "annualized_rate":
            try:
                x = agg_mod.annualized_rate(feats, ages)
            except ValueError:
                continue

        elif aggregation in ("lme_slope", "lme_slope_change"):
            if lme_subjects is None or lme_slopes is None:
                raise RuntimeError("LME output required for lme_slope/lme_slope_change.")
            try:
                x = agg_mod.lme_slope(sid, lme_subjects, lme_slopes)
            except KeyError:
                continue

        elif aggregation == "difference":
            x = agg_mod.difference(feats)

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        rows_X.append(x)
        rows_band.append(mrow["band"])
        rows_sid.append(sid)

    if not rows_X:
        return np.empty((0, 0)), np.array([]), np.array([])

    return (
        np.array(rows_X, dtype=np.float64),
        np.array(rows_band),
        np.array(rows_sid),
    )


# ---------------------------------------------------------------------------
# Option 2 — Feature-wise Welch t-test + BH-FDR
# ---------------------------------------------------------------------------

def run_option2_ttest(
    X: np.ndarray,
    bands: np.ndarray,
    feature_names: list[str] | None = None,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> pd.DataFrame:
    """Per-element Welch t-test (group_a vs group_b) with BH-FDR correction.

    Args:
        X:             (n_subjects, d)
        bands:         (n_subjects,) group labels
        feature_names: optional list of length d
        groups:        (group_a, group_b) — positive direction is group_a > group_b

    Returns DataFrame with columns:
        feature, mean_<group_a>, mean_<group_b>, t_stat, p_raw, p_fdr, cohen_d, significant
    """
    from statsmodels.stats.multitest import multipletests

    group_a, group_b = groups
    Xc = X[bands == group_a]
    Xa = X[bands == group_b]
    d  = X.shape[1]
    if feature_names is None:
        feature_names = generic_names(d)

    t_stats = np.zeros(d)
    p_raw   = np.ones(d)
    cohen_d = np.zeros(d)

    for j in range(d):
        c, a = Xc[:, j], Xa[:, j]
        if len(np.unique(c)) < 2 or len(np.unique(a)) < 2:
            p_raw[j] = np.nan
            continue
        t, p = stats.ttest_ind(c, a, equal_var=False)
        t_stats[j] = float(t)
        p_raw[j]   = float(p)
        sd_pool = np.sqrt((c.std(ddof=1) ** 2 + a.std(ddof=1) ** 2) / 2)
        cohen_d[j] = (c.mean() - a.mean()) / (sd_pool + 1e-10)

    p_fdr = np.full(d, np.nan)
    valid = ~np.isnan(p_raw)
    if valid.sum() > 0:
        _, p_fdr_v, _, _ = multipletests(p_raw[valid], method="fdr_bh")
        p_fdr[valid] = p_fdr_v

    return pd.DataFrame({
        "feature":              feature_names,
        f"mean_{group_a}":      Xc.mean(axis=0),
        f"mean_{group_b}":      Xa.mean(axis=0),
        "t_stat":               t_stats,
        "p_raw":                p_raw,
        "p_fdr":                p_fdr,
        "cohen_d":              cohen_d,
        "significant":          p_fdr < 0.05,
    })


# ---------------------------------------------------------------------------
# Option 3 — Raw-feature baseline: per-subject OLS slope t-test (pre-aggregation)
# ---------------------------------------------------------------------------

def _per_subject_ols_slopes_grouped(
    X_flat: np.ndarray,
    ages_flat: np.ndarray,
    sids_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject OLS slopes using per-subject age+feature centering.

    Returns unique_sids (n_sub,) and slopes (n_sub, n_feat).
    """
    unique_sids = np.unique(sids_flat)
    slopes = np.zeros((len(unique_sids), X_flat.shape[1]), dtype=np.float64)
    for i, sid in enumerate(unique_sids):
        mask = sids_flat == sid
        a  = ages_flat[mask]
        ac = a - a.mean()
        xc = X_flat[mask] - X_flat[mask].mean(axis=0)
        denom = float((ac ** 2).sum())
        if denom > 1e-12:
            slopes[i] = (ac[:, None] * xc).sum(axis=0) / denom
    return unique_sids, slopes


def run_option3_raw_baseline(
    X_flat: np.ndarray,
    ages_flat: np.ndarray,
    sids_flat: np.ndarray,
    bands_flat: np.ndarray,
    feature_names: list[str] | None = None,
    use_statsmodels: bool = False,
    n_jobs: int = 1,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> pd.DataFrame:
    """Welch t-test on per-subject OLS slopes (raw features, before aggregation).

    Serves as a per-extractor ceiling: the maximum group-differentiating
    information available before any aggregation compresses it.

    When use_statsmodels=True (FS ROI only), fits proper MixedLM per feature
    for BLUP slopes instead of plain OLS.
    """
    if use_statsmodels:
        # Use proper StatsmodelsLME for BLUP slopes
        from brainage_agg.agg.lme import StatsmodelsLME
        ages_c = ages_flat - ages_flat.mean()
        lme = StatsmodelsLME()
        print("    Fitting StatsmodelsLME for raw baseline (may take several minutes)...")
        lme.fit(X_flat, ages_c, sids_flat)
        unique_sids, slopes = lme.transform(X_flat, ages_c, sids_flat)
    else:
        unique_sids, slopes = _per_subject_ols_slopes_grouped(X_flat, ages_flat, sids_flat)

    sid_to_band = dict(zip(sids_flat, bands_flat))
    bands_sub = np.array([sid_to_band.get(s, "") for s in unique_sids])

    n_feat = slopes.shape[1]
    if feature_names is None:
        feature_names = generic_names(n_feat)

    return run_option2_ttest(slopes, bands_sub, feature_names, groups=groups)


# ---------------------------------------------------------------------------
# Option 4 — Permutation MANOVA (Hotelling's T² in PCA space)
# ---------------------------------------------------------------------------

def _hotelling_t2(Xc: np.ndarray, Xa: np.ndarray) -> float:
    n1, n2 = len(Xc), len(Xa)
    m1, m2 = Xc.mean(axis=0), Xa.mean(axis=0)
    diff = m1 - m2
    S1 = np.cov(Xc, rowvar=False, ddof=1) if n1 > 1 else np.eye(Xc.shape[1])
    S2 = np.cov(Xa, rowvar=False, ddof=1) if n2 > 1 else np.eye(Xa.shape[1])
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    try:
        Si = np.linalg.inv(Sp + np.eye(Sp.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        Si = np.linalg.pinv(Sp)
    return float((n1 * n2) / (n1 + n2) * (diff @ Si @ diff))


def run_option4_manova(
    X: np.ndarray,
    bands: np.ndarray,
    n_permutations: int = 1000,
    n_pcs: int | None = None,
    seed: int = 0,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> dict:
    """Permutation MANOVA on aggregated vectors.

    Returns dict: observed_t2, p_value, n_pcs, variance_explained,
                  n_<group_a>, n_<group_b>, n_permutations.
    """
    rng = np.random.default_rng(seed)
    group_a, group_b = groups
    mask_a = bands == group_a
    mask_b = bands == group_b
    n_a, n_b = mask_a.sum(), mask_b.sum()

    if n_a < 3 or n_b < 3:
        return {"observed_t2": np.nan, "p_value": np.nan,
                "n_pcs": 0, "variance_explained": np.nan,
                f"n_{group_a}": int(n_a), f"n_{group_b}": int(n_b),
                "n_permutations": n_permutations}

    Xz = StandardScaler().fit_transform(X)
    max_pcs = min(n_a, n_b) - 2
    k = min(max_pcs, 20) if n_pcs is None else min(n_pcs, max_pcs)
    k = min(k, X.shape[1], X.shape[0] - 1)  # PCA constraint: k ≤ min(n_samples-1, n_features)
    k = max(1, k)

    pca = PCA(n_components=k, random_state=seed)
    Xpc = pca.fit_transform(Xz)

    obs_t2 = _hotelling_t2(Xpc[mask_a], Xpc[mask_b])
    null = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(bands)
        null[i] = _hotelling_t2(Xpc[perm == group_a], Xpc[perm == group_b])

    p_value = float((null >= obs_t2).mean())
    return {
        "observed_t2":        float(obs_t2),
        "p_value":             p_value,
        "n_pcs":               k,
        "variance_explained":  float(pca.explained_variance_ratio_.sum()),
        f"n_{group_a}":        int(n_a),
        f"n_{group_b}":        int(n_b),
        "n_permutations":      n_permutations,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_volcano(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    x_col: str = "cohen_d",
    p_col: str = "p_fdr",
    label_top_n: int = 10,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    group_a, group_b = groups
    x = df[x_col].values
    y = -np.log10(df[p_col].clip(lower=1e-300).values)
    sig = df["significant"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[~sig],             y[~sig],             s=8,  alpha=0.3, color="grey",      label="n.s.")
    ax.scatter(x[sig & (x > 0)],   y[sig & (x > 0)],   s=14, alpha=0.8, color="tomato",    label=f"{group_a} > {group_b}")
    ax.scatter(x[sig & (x <= 0)],  y[sig & (x <= 0)],  s=14, alpha=0.8, color="steelblue", label=f"{group_b} > {group_a}")

    # Label the top features by effect size
    top_idx = df["significant"] & (df[x_col].abs() >= df[df["significant"]][x_col].abs().nlargest(label_top_n).min())
    for _, row in df[top_idx].iterrows():
        ax.annotate(
            row["feature"], xy=(row[x_col], -np.log10(max(row[p_col], 1e-300))),
            fontsize=5, ha="center", va="bottom",
        )

    ax.axhline(-np.log10(0.05), color="black", linewidth=0.8, linestyle="--", label="FDR=0.05")
    ax.set_xlabel(f"Cohen's d  (positive = {group_a} higher)")
    ax.set_ylabel("−log₁₀(p_fdr)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path.name}")


def make_summary_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    group_a, group_b = groups
    agg_order = ["mean", "concatenation", "annualized_rate", "lme_slope",
                 "difference", "lme_slope_change"]

    for metric, label, cmap, fmt in [
        ("frac_sig",   "Significant fraction (FDR<0.05)",  "YlOrRd",   ".2f"),
        ("manova_neg_log_p", "−log₁₀(MANOVA p)",          "YlOrRd",   ".1f"),
    ]:
        if metric not in summary.columns:
            continue
        pivot = (
            summary.groupby(["extractor_group", "aggregation"])[metric]
            .mean()
            .reset_index()
            .pivot(index="extractor_group", columns="aggregation", values=metric)
        )
        # Reorder columns
        cols = [c for c in agg_order if c in pivot.columns]
        pivot = pivot[cols]

        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.4), max(4, len(pivot) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    cbar_kws={"label": label})
        ax.set_title(f"{group_a} vs {group_b} group difference — {label}")
        ax.set_ylabel("Extractor group")
        ax.set_xlabel("Aggregation")
        plt.tight_layout()
        fname = f"heatmap_{metric}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Core: process one extractor
# ---------------------------------------------------------------------------

def _extractor_metadata(npz_path: Path, is_roi: bool) -> dict:
    parts = {
        seg.split("-", 1)[0]: seg.split("-", 1)[1]
        for seg in npz_path.stem.split("__")[1:]
        if "-" in seg
    }
    arch = parts.get("model", "double_conv")  # files without model- prefix are double_conv
    return {
        "extractor":      npz_path.stem,
        "extractor_group": "FS ROI" if is_roi
                           else f"{arch} / {parts.get('scaler','?')} / {parts.get('channels','?')}",
        "cnn_arch":       arch,
        "cnn_scaler":     parts.get("scaler", "none"),
        "cnn_channels":   parts.get("channels", "all_roi"),
        "cnn_seed":       int(parts.get("seed", "0")),
    }


def process_one_extractor(
    npz_path: Path,
    manifest: pd.DataFrame,
    out_dir: Path,
    n_permutations: int = 1000,
    n_jobs: int = 1,
    skip_lme_aggs: bool = False,
    groups: tuple[str, str] = ("Child", "Adult"),
) -> list[dict]:
    """Process a single extractor NPZ file. Returns summary rows (one per aggregation).

    Writes per-extractor partial files to out_dir/partial/.
    For FS ROI also writes per-aggregation feature CSVs and volcano plots.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.features.loader import load_features, align_to_manifest, is_fs_roi_file
    from brainage_agg.agg.lme import make_lme_estimator
    from brainage_agg.agg.aggregations import AGGREGATION_SPECS

    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    is_roi = is_fs_roi_file(npz_path)
    meta   = _extractor_metadata(npz_path, is_roi)
    extractor_id = meta["extractor"]

    # --- Load features ---
    X_npz, meta_df, _ = load_features(npz_path)
    subject_data = align_to_manifest(X_npz, meta_df, manifest)
    n_feat = X_npz.shape[1]
    base_names = load_feature_names(npz_path) if is_roi else None

    # --- Build flat observation arrays ---
    flat_X, flat_ages, flat_sids, flat_bands = [], [], [], []
    cohort_col = "cohort_span1" if "cohort_span1" in manifest.columns else "cohort_all"
    eligible = manifest[manifest[cohort_col] & manifest["band"].isin(list(groups))]
    for _, mrow in eligible.iterrows():
        sid = mrow["subject_id"]
        if sid not in subject_data:
            continue
        sd = subject_data[sid]
        for feat, age in zip(sd["features"], sd["ages"]):
            flat_X.append(feat);   flat_ages.append(age)
            flat_sids.append(sid); flat_bands.append(mrow["band"])

    flat_X     = np.array(flat_X,    dtype=np.float64)
    flat_ages  = np.array(flat_ages, dtype=np.float64)
    flat_sids  = np.array(flat_sids)
    flat_bands = np.array(flat_bands)

    # --- Option 3: raw-feature baseline (VectorizedOLS for all extractors) ---
    # StatsmodelsLME would give marginally better BLUP slopes but takes ~7 min
    # for FS ROI and makes the per-extractor runtime inconsistent across the array.
    print(f"  Option 3 (raw baseline)...", end=" ", flush=True)
    opt3_df = run_option3_raw_baseline(
        flat_X, flat_ages, flat_sids, flat_bands,
        feature_names=base_names,
        use_statsmodels=False,
        n_jobs=n_jobs,
        groups=groups,
    )
    n_sig3 = int(opt3_df["significant"].sum())
    print(f"{n_sig3}/{n_feat} sig (FDR<0.05)")

    if is_roi:
        opt3_df.to_csv(out_dir / "option3_raw_baseline_fs_roi.csv", index=False)
        group_a, group_b = groups
        make_volcano(
            opt3_df,
            title=f"FS ROI — raw slope difference ({group_a} vs {group_b}) — Option 3 baseline",
            out_path=out_dir / "volcano_fs_roi_raw_baseline.png",
            groups=groups,
        )
    # Always write a partial option3 summary row
    pd.DataFrame([{**meta, "n_features": n_feat, "n_sig_raw": n_sig3,
                   "frac_sig_raw": n_sig3 / n_feat}]).to_csv(
        partial_dir / f"{extractor_id}_option3.csv", index=False)

    # --- Fit LME (for lme_slope / lme_slope_change aggregations) ---
    ages_c = flat_ages - flat_ages.mean()
    lme_subjects = lme_slopes = None
    if not skip_lme_aggs:
        lme_est = make_lme_estimator(n_feat, threshold=700)
        print(f"  Fitting LME ({type(lme_est).__name__})...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lme_est.fit(flat_X, ages_c, flat_sids)
        lme_subjects, lme_slopes = lme_est.transform(flat_X, ages_c, flat_sids)
        print("done")

    # --- Per-aggregation: Options 2 and 4 ---
    aggregations = [a for a in AGGREGATION_SPECS
                    if not (skip_lme_aggs and AGGREGATION_SPECS[a]["needs_lme"])]

    summary_rows = []
    for aggregation in aggregations:
        try:
            X_agg, bands_agg, _ = build_aggregated_vectors(
                manifest, subject_data, aggregation,
                lme_subjects=lme_subjects, lme_slopes=lme_slopes,
                groups=groups,
            )
        except Exception as e:
            print(f"  SKIP {aggregation}: {e}")
            continue

        if len(X_agg) < 6 or X_agg.shape[1] == 0:
            continue

        d_agg = X_agg.shape[1]
        if aggregation == "concatenation":
            # 3-session: [t_first, t_mid, t_last]; 2-session fallback: [t_first, t_last]
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

        opt2_df = run_option2_ttest(X_agg, bands_agg, agg_feat_names, groups=groups)
        manova  = run_option4_manova(X_agg, bands_agg, n_permutations=n_permutations, groups=groups)

        n_sig2 = int(opt2_df["significant"].sum())
        group_a, group_b = groups
        print(f"  {aggregation:20s}  n_sig={n_sig2:4d}/{d_agg}  "
              f"T²={manova['observed_t2']:7.1f}  p={manova['p_value']:.4f}")

        row = {
            **meta,
            "aggregation":          aggregation,
            "n_total_elements":     d_agg,
            "n_significant":        n_sig2,
            "frac_sig":             n_sig2 / d_agg,
            "manova_t2":            manova["observed_t2"],
            "manova_p":             manova["p_value"],
            "manova_neg_log_p":     -np.log10(max(manova["p_value"], 1e-4)),
            f"n_{group_a}":         int((bands_agg == group_a).sum()),
            f"n_{group_b}":         int((bands_agg == group_b).sum()),
            "n_sig_raw":            n_sig3,
        }
        summary_rows.append(row)

        # Detailed outputs only for FS ROI
        if is_roi:
            opt2_df.to_csv(out_dir / f"features_fs_roi_{aggregation}.csv", index=False)
            make_volcano(
                opt2_df,
                title=f"FS ROI — {aggregation} ({group_a} vs {group_b})",
                out_path=out_dir / f"volcano_fs_roi_{aggregation}.png",
                groups=groups,
            )

    # Write partial summary for this extractor
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            partial_dir / f"{extractor_id}_summary.csv", index=False)

    return summary_rows


# ---------------------------------------------------------------------------
# Merge: collect all partial CSVs → final summary + heatmaps
# ---------------------------------------------------------------------------

def merge(result_dir: Path, groups: tuple[str, str] = ("Child", "Adult")) -> None:
    """Combine all partial extractor results into summary.csv and generate heatmaps."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))

    out_dir     = result_dir / "group_diff"
    partial_dir = out_dir / "partial"

    summary_parts = sorted(partial_dir.glob("*_summary.csv"))
    option3_parts = sorted(partial_dir.glob("*_option3.csv"))

    if not summary_parts:
        print("No partial summary files found — did the array jobs complete?")
        return

    summary_df = pd.concat([pd.read_csv(p) for p in summary_parts], ignore_index=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"summary.csv: {len(summary_df)} rows from {len(summary_parts)} extractors")

    if option3_parts:
        opt3_df = pd.concat([pd.read_csv(p) for p in option3_parts], ignore_index=True)
        opt3_df.to_csv(out_dir / "option3_raw_baseline_summary.csv", index=False)
        print(f"option3_raw_baseline_summary.csv: {len(opt3_df)} rows")

    print("Generating heatmaps...")
    make_summary_heatmaps(summary_df, out_dir, groups=groups)

    print("\n=== Mean frac_sig per extractor_group × aggregation ===")
    tbl = (summary_df.groupby(["extractor_group", "aggregation"])["frac_sig"]
           .mean().unstack("aggregation"))
    print(tbl.to_string())
    print(f"\nAll outputs in {out_dir}/")


# ---------------------------------------------------------------------------
# Sequential run (no SLURM — loops over all extractors in one process)
# ---------------------------------------------------------------------------

def _get_dataset_config(dataset: str, project_root: Path) -> dict:
    """Return paths and group labels for a given dataset."""
    if dataset == "ppmi":
        return {
            "groups":       ("PD", "HC"),
            "result_dir":   project_root / "brainage_agg" / "ppmi_outputs",
            "feature_dir":  project_root / "PPMI_data" / "features",
            "fs_roi_file":  "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz",
        }
    # default: nki
    return {
        "groups":       ("Child", "Adult"),
        "result_dir":   project_root / "brainage_agg" / "outputs",
        "feature_dir":  project_root / "outputs" / "features",
        "fs_roi_file":  "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz",
    }


def run(
    result_dir: Path,
    n_permutations: int = 1000,
    n_jobs: int = 1,
    skip_lme_aggs: bool = False,
    groups: tuple[str, str] = ("Child", "Adult"),
    feature_dir: Path | None = None,
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from brainage_agg.data.manifest import load_manifest
    from brainage_agg.features.loader import list_cnn_files, is_fs_roi_file

    out_dir = result_dir / "group_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parents[2]
    manifest     = load_manifest(result_dir / "manifest.csv")

    if feature_dir is None:
        feature_dir = project_root / "outputs" / "features"
    fs_roi_path  = feature_dir / "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
    all_paths    = [fs_roi_path] + list_cnn_files(feature_dir)

    for i, npz_path in enumerate(all_paths):
        print(f"\n[{i+1}/{len(all_paths)}] {Path(npz_path).name}")
        process_one_extractor(
            Path(npz_path), manifest, out_dir,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            skip_lme_aggs=skip_lme_aggs,
            groups=groups,
        )

    merge(result_dir, groups=groups)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group difference analysis: compare extractor × aggregation.\n\n"
        "Three modes:\n"
        "  (default)         Sequential: process all extractors then merge.\n"
        "  --extractor-idx N SLURM array: process extractor N, write partial files.\n"
        "  --merge           Merge all partial files from a completed array job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["nki", "ppmi"], default="nki",
        help="Dataset to run: 'nki' (Child vs Adult, default) or 'ppmi' (PD vs HC).",
    )
    parser.add_argument("--result-dir",  type=Path, default=None,
                        help="Override result directory (default: auto from --dataset).")
    parser.add_argument("--feature-dir", type=Path, default=None,
                        help="Override feature directory (default: auto from --dataset).")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs",         type=int, default=1,
                        help="Parallel jobs for Option 3 LME fitting (FS ROI only).")
    parser.add_argument("--skip-lme",       action="store_true",
                        help="Skip lme_slope / lme_slope_change aggregations.")
    parser.add_argument("--extractor-idx",  type=int, default=None,
                        help="Process only this extractor index (0-based). "
                             "Used by SLURM array jobs.")
    parser.add_argument("--merge",          action="store_true",
                        help="Merge partial files from a completed SLURM array run.")
    args = parser.parse_args()

    project_root = Path(__file__).parents[2]
    ds_cfg       = _get_dataset_config(args.dataset, project_root)
    groups       = ds_cfg["groups"]
    result_dir   = args.result_dir  if args.result_dir  is not None else ds_cfg["result_dir"]
    feature_dir  = args.feature_dir if args.feature_dir is not None else ds_cfg["feature_dir"]

    if args.merge:
        merge(result_dir, groups=groups)
        return

    if args.extractor_idx is not None:
        # Single-extractor mode for SLURM array tasks
        import sys
        sys.path.insert(0, str(project_root))
        from brainage_agg.data.manifest import load_manifest
        from brainage_agg.features.loader import list_cnn_files

        manifest     = load_manifest(result_dir / "manifest.csv")
        fs_roi_path  = feature_dir / ds_cfg["fs_roi_file"]
        all_paths    = [fs_roi_path] + list_cnn_files(feature_dir)

        idx = args.extractor_idx
        if idx >= len(all_paths):
            print(f"Index {idx} out of range (total: {len(all_paths)}). Nothing to do.")
            return

        out_dir  = result_dir / "group_diff"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = Path(all_paths[idx])
        print(f"Extractor {idx}/{len(all_paths)-1}: {npz_path.name}")
        process_one_extractor(
            npz_path, manifest, out_dir,
            n_permutations=args.n_permutations,
            n_jobs=args.n_jobs,
            skip_lme_aggs=args.skip_lme,
            groups=groups,
        )
        return

    # Sequential fallback
    run(
        result_dir=result_dir,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
        skip_lme_aggs=args.skip_lme,
        groups=groups,
        feature_dir=feature_dir,
    )


if __name__ == "__main__":
    main()
