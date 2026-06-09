"""Ridge regression predicting clinical scores from aggregated brain features (PD only).

Targets and their semantically matched aggregations:
  Severity (static/pooled features):
    NP3TOT_first, NHY_first, MOCA_first, MOCA_last, disease_duration_first
    → cross_sectional, mean, concatenation
  Progression (longitudinal change features):
    NP3TOT_change, NP3TOT_change_rate, NHY_change, MOCA_change
    → annualized_rate, difference, lme_slope_change

All-target mode runs all targets × their matched aggregations × all extractors.
Legacy single-target mode still works for backward compatibility.

Build manifest first:
    python3 -m brainage_agg.data.add_updrs

Usage:
    # All targets, all extractors (recommended):
    python3 -m brainage_agg.experiment.run_updrs --all-targets
    # SLURM array for all targets (one job per extractor):
    python3 -m brainage_agg.experiment.run_updrs --job-id $SLURM_ARRAY_TASK_ID --all-targets
    # Single target, all extractors:
    python3 -m brainage_agg.experiment.run_updrs --all --target NP3TOT_change_rate
    # Merge partial results:
    python3 -m brainage_agg.experiment.run_updrs --merge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from brainage_agg.agg import aggregations as agg_mod
from brainage_agg.agg.lme import VectorizedOLSSlopes
from brainage_agg.features.loader import (
    load_features, align_to_manifest,
    list_cnn_files, is_fs_roi_file,
)
import yaml


UPDRS_AGGS = [
    "cross_sectional",
    "mean",
    "concatenation",
    "annualized_rate",
    "difference",
    "lme_slope_change",
]

# Aggregations semantically matched to each target type.
# Severity targets → static/pooled features; progression targets → change features.
_STATIC_AGGS      = ["cross_sectional", "mean", "concatenation"]
_CHANGE_AGGS      = ["annualized_rate", "difference", "lme_slope_change"]

TARGET_AGGS: dict[str, list[str]] = {
    "NP3TOT_first":           _STATIC_AGGS,
    "NHY_first":              _STATIC_AGGS,
    "MOCA_first":             _STATIC_AGGS,
    "MOCA_last":              _STATIC_AGGS,
    "disease_duration_first": _STATIC_AGGS,
    "NP3TOT_change":          _CHANGE_AGGS,
    "NP3TOT_change_rate":     _CHANGE_AGGS,
    "NHY_change":             _CHANGE_AGGS,
    "MOCA_change":            _CHANGE_AGGS,
}

ALL_TARGETS = list(TARGET_AGGS.keys())

N_FOLDS      = 5
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
T_MID_SEED   = 42


def _load_config(config_path: Path, project_root: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("cohort_csv", "feature_dir"):
        val = cfg.get("data", {}).get(key, "")
        if val and not Path(val).is_absolute():
            cfg["data"][key] = str(project_root / val)
    result_dir = cfg.get("output", {}).get("result_dir", "")
    if result_dir and not Path(result_dir).is_absolute():
        cfg["output"]["result_dir"] = str(project_root / result_dir)
    return cfg


def _flat_lme_inputs(
    sids: list,
    subject_data: dict,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = np.vstack([subject_data[s]["features"] for s in sids])
    ages  = np.concatenate([subject_data[s]["ages"] for s in sids])
    ids   = np.concatenate([[s] * len(subject_data[s]["ages"]) for s in sids])
    return scaler.transform(feats), ages, ids


def _build_regression_matrix(
    manifest_sub: pd.DataFrame,
    subject_data: dict,
    aggregation: str,
    target_col: str,
    lme_subs: np.ndarray | None = None,
    lme_slopes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, subject_ids) for Ridge regression on NP3TOT.

    Only includes subjects with a valid (non-NaN) target score.
    """
    rows_X, rows_y, groups = [], [], []
    sid_manifest = manifest_sub.set_index("subject_id")

    for sid in manifest_sub["subject_id"]:
        if sid not in subject_data:
            continue
        mrow  = sid_manifest.loc[sid]
        y_val = mrow[target_col]
        if pd.isna(y_val):
            continue

        data  = subject_data[sid]
        feats = data["features"]
        ages  = data["ages"]

        if aggregation == "cross_sectional":
            x = feats[0]
        elif aggregation == "mean":
            x = agg_mod.mean(feats)
        elif aggregation == "concatenation":
            if mrow["t_mid_idx"] < 0:
                x = np.concatenate([feats[0], feats[-1]])
            else:
                t_mid_local = list(mrow["row_indices"]).index(mrow["t_mid_idx"])
                x = agg_mod.concatenation(feats, 0, t_mid_local, len(feats) - 1)
        elif aggregation == "annualized_rate":
            dt = float(ages[-1] - ages[0])
            if abs(dt) < 1e-8:
                continue
            x = agg_mod.annualized_rate(feats, ages)
        elif aggregation == "difference":
            x = agg_mod.difference(feats)
        elif aggregation == "lme_slope_change":
            if lme_subs is None:
                continue
            x = agg_mod.lme_slope(sid, lme_subs, lme_slopes)
        else:
            continue

        rows_X.append(x)
        rows_y.append(float(y_val))
        groups.append(sid)

    if not rows_X:
        return np.empty((0, 0)), np.empty(0), np.empty(0, dtype=object)
    return (
        np.array(rows_X, dtype=np.float32),
        np.array(rows_y, dtype=np.float32),
        np.array(groups),
    )


def run_updrs_for_file(
    npz_path: Path,
    manifest: pd.DataFrame,
    target_col: str = "NP3TOT_first",
    n_folds: int = N_FOLDS,
    ridge_alphas: list[float] = RIDGE_ALPHAS,
    agg_list: list[str] | None = None,
    perm_seed: int | None = None,
) -> list[dict]:
    """Run aggregations × folds for one feature file. Returns per-fold rows.

    agg_list restricts which aggregations are run; defaults to TARGET_AGGS[target_col]
    if defined, else all UPDRS_AGGS.

    perm_seed: if not None, shuffles the target column at subject level before CV
               (for permutation null distribution).
    """
    from brainage_agg.data.manifest import load_manifest as _lm

    if agg_list is None:
        agg_list = TARGET_AGGS.get(target_col, UPDRS_AGGS)

    X_raw, meta_df, file_meta = load_features(npz_path)
    subject_data = align_to_manifest(X_raw, meta_df, manifest)

    extractor_type = "fs_roi" if is_fs_roi_file(npz_path) else "cnn"
    extractor_id   = npz_path.stem

    rows = []
    for aggregation in agg_list:
        spec = agg_mod.AGGREGATION_SPECS.get(aggregation, {})

        cohort_col = "cohort_span1" if "cohort_span1" in manifest.columns else "cohort_all"
        if spec.get("cohort") == "concat":
            mdf_c = manifest[manifest["cohort_concat"]].copy()
            mdf = mdf_c if len(mdf_c) > 0 else manifest[manifest[cohort_col]].copy()
        else:
            mdf = manifest[manifest[cohort_col]].copy()

        if spec.get("needs_ages") or aggregation in ("difference", "lme_slope_change"):
            mdf = mdf[~mdf["exclude_arm_b"]]

        # For cross_sectional, no longitudinal requirement — apply span filter when present
        if aggregation == "cross_sectional":
            if "cohort_span1" in manifest.columns:
                mdf = manifest[manifest["cohort_span1"]].copy()
            else:
                mdf = manifest.copy()

        # Only PD subjects with valid NP3TOT (filtering is done inside _build_regression_matrix)
        sids_all = [s for s in mdf["subject_id"] if s in subject_data]
        if len(sids_all) < n_folds * 3:
            continue

        # Permutation null: shuffle target values at subject level before CV
        if perm_seed is not None:
            rng = np.random.default_rng(perm_seed + hash(target_col + aggregation) % (2**31))
            mdf = mdf.copy()
            sid_rows = mdf["subject_id"].isin(sids_all)
            unique_sids = list(dict.fromkeys(mdf.loc[sid_rows, "subject_id"]))
            y_vals = np.array([
                mdf.loc[mdf["subject_id"] == s, target_col].iloc[0]
                for s in unique_sids
            ], dtype=float)
            shuffled_y = rng.permutation(y_vals)
            sid_to_y = dict(zip(unique_sids, shuffled_y))
            mdf.loc[sid_rows, target_col] = mdf.loc[sid_rows, "subject_id"].map(sid_to_y)

        # GroupKFold splits by subject (no leakage across timepoints)
        dummy_X = np.zeros((len(sids_all), 1))
        gkf = GroupKFold(n_splits=n_folds)

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(dummy_X, groups=sids_all)):
            tr_sids = [sids_all[i] for i in tr_idx]
            te_sids = [sids_all[i] for i in te_idx]

            X_tr_flat = np.vstack([subject_data[s]["features"] for s in tr_sids])
            scaler    = StandardScaler().fit(X_tr_flat)

            sd_scaled = {
                s: {
                    "features":    scaler.transform(subject_data[s]["features"]),
                    "ages":        subject_data[s]["ages"],
                    "row_indices": subject_data[s].get("row_indices", []),
                }
                for s in set(tr_sids + te_sids)
            }

            lme_subs = lme_slopes = None
            if aggregation == "lme_slope_change":
                lme_feats, lme_ages, lme_ids = _flat_lme_inputs(tr_sids, subject_data, scaler)
                lme_model = VectorizedOLSSlopes().fit(lme_feats, lme_ages, lme_ids)
                all_sids  = list(dict.fromkeys(tr_sids + te_sids))
                a_feats, a_ages, a_ids = _flat_lme_inputs(all_sids, subject_data, scaler)
                lme_subs, lme_slopes  = lme_model.transform(a_feats, a_ages, a_ids)

            tr_mdf = mdf[mdf["subject_id"].isin(tr_sids)]
            te_mdf = mdf[mdf["subject_id"].isin(te_sids)]

            X_tr, y_tr, _ = _build_regression_matrix(
                tr_mdf, sd_scaled, aggregation, target_col, lme_subs, lme_slopes,
            )
            X_te, y_te, _ = _build_regression_matrix(
                te_mdf, sd_scaled, aggregation, target_col, lme_subs, lme_slopes,
            )

            if len(X_tr) < 4 or len(y_tr) < 2 or len(X_te) < 2:
                continue

            # Inner CV for alpha selection
            ridge = GridSearchCV(
                Ridge(), {"alpha": ridge_alphas},
                cv=min(3, len(X_tr)),
                scoring="neg_mean_absolute_error",
            )
            ridge.fit(X_tr, y_tr)
            best_alpha = float(ridge.best_params_["alpha"])
            y_pred = ridge.predict(X_te)

            mae  = float(np.mean(np.abs(y_te - y_pred)))
            rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            r    = float(spearmanr(y_te, y_pred)[0]) if len(y_te) > 2 else 0.0
            chance_mae = float(np.mean(np.abs(y_te - y_tr.mean())))

            rows.append({
                "target":        target_col,
                "extractor_type": extractor_type,
                "extractor":      extractor_id,
                "cnn_arch":       file_meta.get("cnn_arch", ""),
                "scaler":         file_meta.get("scaler", ""),
                "channels":       file_meta.get("channels", ""),
                "cnn_seed":       file_meta.get("seed", ""),
                "aggregation":    aggregation,
                "fold":           fold,
                "MAE":            mae,
                "RMSE":           rmse,
                "R2":             r2,
                "r":              r,
                "chance_mae":     chance_mae,
                "best_alpha":     best_alpha,
                "n_train":        int(len(y_tr)),
                "n_test":         int(len(y_te)),
            })

    return rows


def enumerate_extractors(cfg: dict) -> list[Path]:
    feature_dir = Path(cfg["data"]["feature_dir"])
    fs_roi      = feature_dir / cfg["data"]["fs_roi_file"]
    cnn         = sorted(list_cnn_files(feature_dir))
    return [fs_roi] + cnn


def merge(result_dir: Path) -> None:
    job_dir = result_dir / "updrs_jobs"
    parts   = sorted(job_dir.glob("*.csv"))
    if not parts:
        print("No partial files found in updrs_jobs/")
        return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = result_dir / "updrs_results.csv"
    df.to_csv(out, index=False)
    print(f"Merged {len(parts)} files → {out}  ({len(df)} rows)")
    print("\nMean r per target × aggregation (FS ROI):")
    fs = df[df["extractor_type"] == "fs_roi"]
    if not fs.empty:
        print(fs.groupby(["target", "aggregation"])["r"].mean().round(3).to_string())


def run_permutation_null(
    cfg: dict,
    manifest: pd.DataFrame,
    n_permutations: int,
    result_dir: Path,
) -> None:
    """Run permutation null for FS ROI extractor across all targets × aggregations.

    Shuffles target labels at subject level n_permutations times; saves
    per-permutation mean r (across folds) to updrs_perm_null.csv.
    Used to establish the 95th-percentile chance threshold for T2 bar plots.
    """
    fs_roi_path = Path(cfg["data"]["feature_dir"]) / cfg["data"]["fs_roi_file"]
    out_path = result_dir / "updrs_perm_null.csv"

    all_rows = []
    for perm_i in range(n_permutations):
        print(f"  Permutation {perm_i+1}/{n_permutations} ...", flush=True)
        for target in ALL_TARGETS:
            rows = run_updrs_for_file(
                fs_roi_path, manifest, target_col=target, perm_seed=perm_i,
            )
            # Aggregate r across folds: mean per (target, aggregation)
            if not rows:
                continue
            df_perm = pd.DataFrame(rows)
            for agg, grp in df_perm.groupby("aggregation"):
                all_rows.append({
                    "perm_seed":   perm_i,
                    "target":      target,
                    "aggregation": agg,
                    "r_mean":      float(grp["r"].mean()),
                    "n_folds":     len(grp),
                })

    df_null = pd.DataFrame(all_rows)
    df_null.to_csv(out_path, index=False)
    print(f"Saved permutation null ({n_permutations} permutations) → {out_path}")
    print("\n95th-percentile chance threshold (r) per target × aggregation:")
    if not df_null.empty:
        r95 = (
            df_null.groupby(["target", "aggregation"])["r_mean"]
            .quantile(0.95)
            .reset_index()
            .rename(columns={"r_mean": "r_chance_95"})
        )
        print(r95.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical score Ridge regression array job.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--job-id", type=int,
                     help="0-indexed extractor file index (SLURM_ARRAY_TASK_ID).")
    grp.add_argument("--all",        action="store_true",
                     help="Run all extractors sequentially for the selected target(s).")
    grp.add_argument("--all-targets", action="store_true",
                     help="Run all extractors × all configured targets (recommended).")
    grp.add_argument("--merge", action="store_true",
                     help="Merge partial job CSVs into updrs_results.csv.")
    grp.add_argument("--list",  action="store_true",
                     help="Print total number of jobs and exit.")
    grp.add_argument("--permute", action="store_true",
                     help="Run permutation null for FS ROI across all targets; save to updrs_perm_null.csv.")

    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).parents[2])
    parser.add_argument("--config",       type=Path, default=None)
    parser.add_argument("--target",       type=str,  default="NP3TOT_first",
                        help="Single target column (used with --all / --job-id).")
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="Number of permutations for --permute mode (default: 200).")
    args = parser.parse_args()

    project_root = args.project_root
    config_path  = args.config or project_root / "brainage_agg" / "ppmi_config.yaml"
    cfg          = _load_config(config_path, project_root)
    result_dir   = Path(cfg["output"]["result_dir"])

    if args.merge:
        merge(result_dir)
        return

    # Load manifest
    # (shared by all modes below)
    manifest_path = result_dir / "manifest_with_updrs.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run: python3 -m brainage_agg.data.add_updrs"
        )
    manifest = pd.read_csv(manifest_path)
    import json as _json
    for col in ("ages", "row_indices"):
        if col in manifest.columns:
            manifest[col] = manifest[col].apply(
                lambda v: _json.loads(v) if isinstance(v, str) else v
            )
    manifest = manifest[manifest["band"] == "PD"].reset_index(drop=True)
    print(f"PD subjects: {len(manifest)}")

    all_paths = enumerate_extractors(cfg)

    if args.list:
        print(f"Total extractors: {len(all_paths)}")
        print(f"Configured targets: {ALL_TARGETS}")
        return

    if args.permute:
        print(f"Running permutation null ({args.n_permutations} permutations, FS ROI only) ...")
        run_permutation_null(cfg, manifest, args.n_permutations, result_dir)
        return

    job_dir = result_dir / "updrs_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)

    if args.all_targets:
        # Run all targets × all extractors sequentially
        for i, npz_path in enumerate(all_paths):
            all_rows = []
            for target in ALL_TARGETS:
                rows = run_updrs_for_file(npz_path, manifest, target_col=target)
                all_rows.extend(rows)
            print(f"[{i+1}/{len(all_paths)}] {npz_path.name}: {len(all_rows)} rows")
            out = job_dir / f"job_{i:04d}.csv"
            pd.DataFrame(all_rows).to_csv(out, index=False)
        merge(result_dir)
        return

    if args.all:
        for i, npz_path in enumerate(all_paths):
            print(f"[{i+1}/{len(all_paths)}] {npz_path.name}")
            rows = run_updrs_for_file(npz_path, manifest, target_col=args.target)
            out  = job_dir / f"job_{i:04d}.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  → {len(rows)} rows → {out.name}")
        merge(result_dir)
        return

    # Single SLURM array job: one extractor, all configured targets
    idx = args.job_id
    if idx >= len(all_paths):
        print(f"Index {idx} out of range (total: {len(all_paths)}). Nothing to do.")
        return
    npz_path = all_paths[idx]
    print(f"Job {idx}/{len(all_paths)-1}: {npz_path.name}")
    all_rows = []
    for target in ALL_TARGETS:
        rows = run_updrs_for_file(npz_path, manifest, target_col=target)
        all_rows.extend(rows)
    out = job_dir / f"job_{idx:04d}.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"  → {len(all_rows)} rows → {out.name}")


if __name__ == "__main__":
    main()
