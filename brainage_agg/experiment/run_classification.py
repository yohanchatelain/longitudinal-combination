"""Band-pair classification experiment (Child/Adult for NKI, PD/HC for PPMI).

Parallelised by extractor file — one SLURM array job per file.

Each job:
  1. Loads a single feature .npz file (FS ROI or CNN variant)
  2. Runs all 6 aggregations × 2 classifiers × 5-fold GroupKFold CV
  3. Writes per-job results to <result_dir>/classif_jobs/job_{N:04d}.csv

Flags:
  --config PATH          config YAML (default: project_root/brainage_agg/config.yaml)
  --positive-class BAND  band label for positive class (default: Adult; use PD for PPMI)

Usage
-----
  # List all jobs (print extractor index → file mapping)
  python run_classification.py --list

  # Run one job locally (extractor index 0 = FS ROI)
  python run_classification.py --job-id 0

  # SLURM array (N_JOBS set by --list)
  sbatch brainage_agg/experiment/submit_classification.sh

  # Merge results after all jobs finish
  python run_classification.py --merge
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2]))

from brainage_agg.agg import aggregations as agg_mod
from brainage_agg.agg.lme import VectorizedOLSSlopes
from brainage_agg.data.manifest import build_manifest
from brainage_agg.features.loader import (
    align_to_manifest,
    is_fs_roi_file,
    list_cnn_files,
    load_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIF_AGGS = [
    "cross_sectional",
    "mean",
    "concatenation",
    "annualized_rate",
    "difference",
    "lme_slope_change",
]

CLASSIFIERS = {
    "LogisticReg": lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"
    ),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=1,
    ),
}

N_FOLDS = 5
T_MID_SEED = 42


def _load_config(config_path: Path, project_root: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths
    for key in ("cohort_csv", "feature_dir"):
        val = cfg.get("data", {}).get(key, "")
        if val and not Path(val).is_absolute():
            cfg["data"][key] = str(project_root / val)
    result_dir = cfg.get("output", {}).get("result_dir", "")
    if result_dir and not Path(result_dir).is_absolute():
        cfg["output"]["result_dir"] = str(project_root / result_dir)
    return cfg


# ---------------------------------------------------------------------------
# Feature matrix builders
# ---------------------------------------------------------------------------

def _flat_lme_inputs(
    sids: list[str],
    subject_data: dict,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack all timepoints for a list of subjects (for LME fitting)."""
    feats = np.vstack([subject_data[s]["features"] for s in sids])
    ages  = np.concatenate([subject_data[s]["ages"]     for s in sids])
    ids   = np.concatenate([[s] * len(subject_data[s]["ages"]) for s in sids])
    return scaler.transform(feats), ages, ids


def _build_subject_matrix(
    manifest_sub: pd.DataFrame,
    subject_data: dict,
    aggregation: str,
    negative_class: str,
    positive_class: str,
    lme_subs: np.ndarray | None = None,
    lme_slopes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per subject: returns (X, y, group_ids)."""
    rows_X, rows_y, groups = [], [], []
    valid_bands = {negative_class, positive_class}

    for _, row in manifest_sub.iterrows():
        sid  = row["subject_id"]
        if sid not in subject_data:
            continue
        band = row["band"]
        if band not in valid_bands:
            continue

        data  = subject_data[sid]
        feats = data["features"]   # already scaled by caller
        ages  = data["ages"]
        y_val = 0 if band == negative_class else 1

        if aggregation == "cross_sectional":
            x = feats[0]
        elif aggregation == "mean":
            x = agg_mod.mean(feats)
        elif aggregation == "concatenation":
            if row["t_mid_idx"] < 0:
                x = np.concatenate([feats[0], feats[-1]])
            else:
                t_mid_local = list(row["row_indices"]).index(row["t_mid_idx"])
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
        rows_y.append(y_val)
        groups.append(sid)

    if not rows_X:
        return np.empty((0, 0)), np.empty(0, dtype=int), np.empty(0, dtype=object)
    return (
        np.array(rows_X, dtype=np.float32),
        np.array(rows_y, dtype=int),
        np.array(groups),
    )


# ---------------------------------------------------------------------------
# Per-extractor CV loop
# ---------------------------------------------------------------------------

def run_classification_for_file(
    npz_path: Path,
    manifest: pd.DataFrame,
    negative_class: str,
    positive_class: str,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """Run all aggregations × classifiers × folds for one feature file."""
    X_raw, meta_df, file_meta = load_features(npz_path)
    subject_data = align_to_manifest(X_raw, meta_df, manifest)

    extractor_type = "fs_roi" if is_fs_roi_file(npz_path) else "cnn"
    extractor_id   = npz_path.stem
    valid_bands    = {negative_class, positive_class}

    rows = []
    for aggregation in CLASSIF_AGGS:
        spec   = agg_mod.AGGREGATION_SPECS.get(aggregation, {})
        cohort_col = "cohort_span1" if "cohort_span1" in manifest.columns else "cohort_all"
        if spec.get("cohort") == "concat":
            primary = manifest[manifest["cohort_concat"] & manifest["band"].isin(valid_bands)]
            mdf = primary if not primary.empty else manifest[manifest[cohort_col] & manifest["band"].isin(valid_bands)]
        else:
            mdf = manifest[manifest[cohort_col] & manifest["band"].isin(valid_bands)]
        mdf = mdf.copy()
        # Change aggregations and annualized_rate require delta_t > 0
        if spec.get("needs_ages") or aggregation in ("difference", "lme_slope_change"):
            mdf = mdf[~mdf["exclude_arm_b"]]

        sids_all = [
            r.subject_id
            for r in mdf.itertuples()
            if r.subject_id in subject_data and r.band in valid_bands
        ]
        if len(sids_all) < n_folds * 4:
            continue

        dummy_X = np.zeros((len(sids_all), 1))
        gkf = GroupKFold(n_splits=n_folds)

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(dummy_X, groups=sids_all)):
            tr_sids = [sids_all[i] for i in tr_idx]
            te_sids = [sids_all[i] for i in te_idx]

            # Z-score on train timepoints only
            X_tr_flat = np.vstack([subject_data[s]["features"] for s in tr_sids])
            scaler    = StandardScaler().fit(X_tr_flat)

            sd_scaled = {
                s: {
                    "features":    scaler.transform(subject_data[s]["features"]),
                    "ages":        subject_data[s]["ages"],
                    "row_indices": subject_data[s]["row_indices"],
                }
                for s in set(tr_sids + te_sids)
            }

            # Fit LME on train, transform train+test
            lme_subs = lme_slopes = None
            if aggregation == "lme_slope_change":
                lme_feats, lme_ages, lme_ids = _flat_lme_inputs(tr_sids, subject_data, scaler)
                lme_model = VectorizedOLSSlopes().fit(lme_feats, lme_ages, lme_ids)
                all_sids  = list(dict.fromkeys(tr_sids + te_sids))
                a_feats, a_ages, a_ids = _flat_lme_inputs(all_sids, subject_data, scaler)
                lme_subs, lme_slopes  = lme_model.transform(a_feats, a_ages, a_ids)

            tr_mdf = mdf[mdf["subject_id"].isin(tr_sids)]
            te_mdf = mdf[mdf["subject_id"].isin(te_sids)]

            X_tr, y_tr, _ = _build_subject_matrix(
                tr_mdf, sd_scaled, aggregation, negative_class, positive_class,
                lme_subs, lme_slopes,
            )
            X_te, y_te, _ = _build_subject_matrix(
                te_mdf, sd_scaled, aggregation, negative_class, positive_class,
                lme_subs, lme_slopes,
            )

            if (len(X_tr) < 4
                    or len(np.unique(y_tr)) < 2
                    or len(np.unique(y_te)) < 2):
                continue

            for clf_name, clf_factory in CLASSIFIERS.items():
                clf  = clf_factory()
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)
                prob = clf.predict_proba(X_te)[:, 1]

                rows.append(
                    {
                        "extractor_type":    extractor_type,
                        "extractor":         extractor_id,
                        "cnn_arch":          file_meta.get("cnn_arch", ""),
                        "scaler":            file_meta.get("scaler", ""),
                        "channels":          file_meta.get("channels", ""),
                        "cnn_seed":          file_meta.get("seed", ""),
                        "aggregation":       aggregation,
                        "classifier":        clf_name,
                        "fold":              fold,
                        "bacc":              float(balanced_accuracy_score(y_te, pred)),
                        "auc":              float(roc_auc_score(y_te, prob)),
                        "f1_macro":          float(f1_score(y_te, pred, average="macro")),
                        "n_train":           int(len(y_tr)),
                        "n_test":            int(len(y_te)),
                        "n_neg_train":       int((y_tr == 0).sum()),
                        "n_pos_train":       int((y_tr == 1).sum()),
                        "negative_class":    negative_class,
                        "positive_class":    positive_class,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Extractor enumeration
# ---------------------------------------------------------------------------

def enumerate_extractors(cfg: dict) -> list[Path]:
    """Return all feature .npz files: [fs_roi] + sorted CNN files."""
    feature_dir = Path(cfg["data"]["feature_dir"])
    fs_roi = feature_dir / cfg["data"]["fs_roi_file"]
    cnn    = sorted(list_cnn_files(feature_dir))
    return [fs_roi] + cnn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Band-pair classification SLURM array job.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--job-id", type=int,
                     help="0-indexed extractor file index (SLURM_ARRAY_TASK_ID).")
    grp.add_argument("--list",   action="store_true",
                     help="Print total number of jobs and exit.")
    grp.add_argument("--merge",  action="store_true",
                     help="Merge all per-job CSVs into classif_results.csv.")

    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).parents[2])
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Config YAML (default: <project-root>/brainage_agg/config.yaml)",
    )
    parser.add_argument(
        "--positive-class", type=str, default="Adult",
        help="Band label for the positive class (default: Adult; use PD for PPMI)",
    )
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    project_root = args.project_root
    config_path  = args.config or (project_root / "brainage_agg" / "config.yaml")
    cfg          = _load_config(config_path, project_root)

    band_map      = cfg.get("data", {}).get("band_map") or {}
    band_values   = list(band_map.values()) if band_map else ["Child", "Adult"]
    positive_class = args.positive_class
    neg_candidates = [b for b in band_values if b != positive_class]
    if not neg_candidates:
        # fallback: use whatever band_map provides besides the positive class
        neg_candidates = ["Child"] if positive_class == "Adult" else ["HC"]
    negative_class = neg_candidates[0]

    out_dir  = Path(cfg["output"]["result_dir"])
    jobs_dir = out_dir / "classif_jobs"

    extractors = enumerate_extractors(cfg)

    # ── --list ──────────────────────────────────────────────────────────────
    if args.list:
        print(len(extractors))
        for i, p in enumerate(extractors):
            print(f"  {i:4d}  {p.name}")
        return

    # ── --merge ─────────────────────────────────────────────────────────────
    if args.merge:
        csvs = sorted(jobs_dir.glob("job_*.csv"))
        if not csvs:
            print("No job CSVs found in", jobs_dir)
            return
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        out_path = out_dir / "classif_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Merged {len(csvs)} files → {len(df)} rows → {out_path}")
        return

    # ── --job-id N ──────────────────────────────────────────────────────────
    job_id = args.job_id
    if job_id < 0 or job_id >= len(extractors):
        print(f"job-id {job_id} out of range [0, {len(extractors)-1}]", file=sys.stderr)
        sys.exit(1)

    npz_path = extractors[job_id]
    print(f"[job {job_id}/{len(extractors)-1}] {npz_path.name}")
    print(f"Classification: {negative_class}(0) vs {positive_class}(1)")

    fs_roi_npz  = extractors[0]
    t_mid_seed  = cfg.get("seeds", {}).get("t_mid", T_MID_SEED)
    band_map_cfg = cfg.get("data", {}).get("band_map")
    manifest    = build_manifest(fs_roi_npz, t_mid_seed=t_mid_seed, band_map=band_map_cfg)

    t0   = time.time()
    rows = run_classification_for_file(
        npz_path, manifest,
        negative_class=negative_class,
        positive_class=positive_class,
        n_folds=args.n_folds,
    )
    elapsed = time.time() - t0

    if not rows:
        print("  No results produced — check cohort / feature alignment.")
        return

    jobs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = jobs_dir / f"job_{job_id:04d}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  {len(rows)} rows → {out_csv}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
