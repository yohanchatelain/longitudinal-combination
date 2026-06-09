"""G1 vs G2 span classification experiment — two feature variants.

**Difference variant** (original):
  G1 (label=1): f_last − f_first  (full-span difference)
  G2 (label=0): f_mid  − f_first  (short-span difference, first → mid)

**Concatenation variant** (new):
  G1 (label=1): [f_first, f_mid, f_last]  (full-span concatenation)
  G2 (label=0): [f_first, f_mid, f_mid]   (short-span: last slot = mid)

Both variants ask the same question: can a classifier tell whether features
encode the full longitudinal span or only the first portion?  Near-chance
(AUC ≈ 0.5) implies the extra time produces no new signal in feature space.
The concatenation variant tests this on absolute feature values rather than
change vectors, so it is immune to catastrophic cancellation.

Design notes
------------
- Only subjects in cohort_concat (≥3 timepoints) are used.
- The same t_mid from the manifest (fixed seed) is used for G2 to ensure
  consistency with the concatenation aggregation.
- GroupKFold on subject_id guarantees both G1 and G2 samples from the same
  subject land in the same fold — no within-subject leakage.
- Runs separately for each band (all / Child / Adult) to avoid confounding
  age-group differences with span-type differences.
- Both variants run in the same SLURM job; output rows carry a `variant`
  column ("difference" or "concatenation").
- Parallelised by extractor file — one SLURM array task per file.

Usage
-----
  python run_span_classification.py --list
  python run_span_classification.py --job-id 0
  python run_span_classification.py --merge
  sbatch brainage_agg/slurm/submit_span_classif.sh
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2]))

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

CLASSIFIERS = {
    "LogisticReg": lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"
    ),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=1,
    ),
}

N_FOLDS    = 5
T_MID_SEED = 42
BANDS      = ("all", "Child", "Adult")


# ---------------------------------------------------------------------------
# Per-fold dataset builder
# ---------------------------------------------------------------------------

def _build_span_matrix(
    sids: list[str],
    mdf: pd.DataFrame,
    scaled_data: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, groups) for the G1/G2 paired dataset.

    For each subject:
      G1 (y=1): f_last − f_first
      G2 (y=0): f_mid  − f_first

    Returns arrays of shape (2 * n_subjects, n_features) (perfectly balanced).
    """
    rows_X, rows_y, rows_g = [], [], []
    sid_manifest = mdf.set_index("subject_id")

    for sid in sids:
        if sid not in scaled_data or sid not in sid_manifest.index:
            continue
        mrow  = sid_manifest.loc[sid]
        feats = scaled_data[sid]["features"]  # (n_tp, n_feat) scaled

        global_row_indices = np.asarray(
            mrow["row_indices"] if isinstance(mrow["row_indices"], list)
            else list(mrow["row_indices"])
        )
        t_mid_global = int(mrow["t_mid_idx"])
        loc = np.where(global_row_indices == t_mid_global)[0]
        if len(loc) == 0:
            continue
        t_mid_local   = int(loc[0])
        t_first_local = 0
        t_last_local  = len(feats) - 1

        if t_mid_local == t_first_local or t_mid_local == t_last_local:
            continue  # degenerate: mid coincides with endpoint

        # G1: full span
        rows_X.append(feats[t_last_local] - feats[t_first_local])
        rows_y.append(1)
        rows_g.append(sid)

        # G2: randomly either first-half (mid−first) or second-half (last−mid)
        # per-subject seed keeps the choice stable across train/test splits
        use_first_half = np.random.default_rng(abs(hash(str(sid))) % (2**31)).integers(2)
        if use_first_half:
            rows_X.append(feats[t_mid_local] - feats[t_first_local])
        else:
            rows_X.append(feats[t_last_local] - feats[t_mid_local])
        rows_y.append(0)
        rows_g.append(sid)

    if not rows_X:
        return (np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty(0, dtype=object))

    return (
        np.array(rows_X, dtype=np.float32),
        np.array(rows_y,  dtype=int),
        np.array(rows_g),
    )


def _build_concat_span_matrix(
    sids: list[str],
    mdf: pd.DataFrame,
    scaled_data: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, groups) for the concatenation-based G1/G2 dataset.

    For each subject:
      G1 (y=1): [f_first, f_mid, f_last]  — full-span concatenation
      G2 (y=0): [f_first, f_mid, f_mid]   — short-span (last slot = mid)

    Returns arrays of shape (2 * n_subjects, 3 * n_features).
    """
    rows_X, rows_y, rows_g = [], [], []
    sid_manifest = mdf.set_index("subject_id")

    for sid in sids:
        if sid not in scaled_data or sid not in sid_manifest.index:
            continue
        mrow  = sid_manifest.loc[sid]
        feats = scaled_data[sid]["features"]

        global_row_indices = np.asarray(
            mrow["row_indices"] if isinstance(mrow["row_indices"], list)
            else list(mrow["row_indices"])
        )
        t_mid_global = int(mrow["t_mid_idx"])
        loc = np.where(global_row_indices == t_mid_global)[0]
        if len(loc) == 0:
            continue
        t_mid_local   = int(loc[0])
        t_first_local = 0
        t_last_local  = len(feats) - 1

        if t_mid_local == t_first_local or t_mid_local == t_last_local:
            continue

        f_first = feats[t_first_local]
        f_mid   = feats[t_mid_local]
        f_last  = feats[t_last_local]

        # G1: full-span — [f_first, f_mid, f_last]
        rows_X.append(np.concatenate([f_first, f_mid, f_last]))
        rows_y.append(1)
        rows_g.append(sid)

        # G2: same random half as the difference variant (same per-subject seed)
        #   first-half → [f_first, f_mid, f_mid]  (window ends at t_mid)
        #   second-half → [f_mid, f_mid, f_last]   (window starts at t_mid)
        use_first_half = np.random.default_rng(abs(hash(str(sid))) % (2**31)).integers(2)
        if use_first_half:
            rows_X.append(np.concatenate([f_first, f_mid, f_mid]))
        else:
            rows_X.append(np.concatenate([f_mid, f_mid, f_last]))
        rows_y.append(0)
        rows_g.append(sid)

    if not rows_X:
        return (np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty(0, dtype=object))

    return (
        np.array(rows_X, dtype=np.float32),
        np.array(rows_y,  dtype=int),
        np.array(rows_g),
    )


# Map variant name → matrix builder
_MATRIX_BUILDERS: dict[str, callable] = {
    "difference":    _build_span_matrix,
    "concatenation": _build_concat_span_matrix,
}


# ---------------------------------------------------------------------------
# Per-extractor loop
# ---------------------------------------------------------------------------

def run_span_classification_for_file(
    npz_path: Path,
    manifest: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """Run G1/G2 classification for all bands × classifiers × folds."""
    X_raw, meta_df, file_meta = load_features(npz_path)
    subject_data = align_to_manifest(X_raw, meta_df, manifest)

    extractor_type = "fs_roi" if is_fs_roi_file(npz_path) else "cnn"
    extractor_id   = npz_path.stem

    rows = []

    for band in BANDS:
        # cohort_concat: ≥3 timepoints (required for a mid-point to exist)
        mdf = manifest[manifest["cohort_concat"]].copy()
        if band != "all":
            mdf = mdf[mdf["band"] == band]
        # delta_t > 0 guard (same as Arm B exclusion)
        mdf = mdf[~mdf["exclude_arm_b"]]

        sids = [r.subject_id for r in mdf.itertuples() if r.subject_id in subject_data]

        if len(sids) < n_folds * 4:
            continue

        gkf     = GroupKFold(n_splits=n_folds)
        dummy_X = np.zeros((len(sids), 1))

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(dummy_X, groups=sids)):
            tr_sids = [sids[i] for i in tr_idx]
            te_sids = [sids[i] for i in te_idx]

            # Z-score fitted on training timepoints only
            X_tr_flat = np.vstack([subject_data[s]["features"] for s in tr_sids])
            scaler    = StandardScaler().fit(X_tr_flat)

            scaled = {
                s: {
                    "features":    scaler.transform(subject_data[s]["features"]),
                    "ages":        subject_data[s]["ages"],
                    "row_indices": subject_data[s]["row_indices"],
                }
                for s in set(tr_sids) | set(te_sids)
            }

            tr_mdf = mdf[mdf["subject_id"].isin(tr_sids)]
            te_mdf = mdf[mdf["subject_id"].isin(te_sids)]

            for variant, build_fn in _MATRIX_BUILDERS.items():
                X_tr, y_tr, _ = build_fn(tr_sids, tr_mdf, scaled)
                X_te, y_te, _ = build_fn(te_sids, te_mdf, scaled)

                if (len(X_tr) < 4
                        or len(np.unique(y_tr)) < 2
                        or len(np.unique(y_te)) < 2):
                    continue

                for clf_name, clf_factory in CLASSIFIERS.items():
                    clf  = clf_factory()
                    clf.fit(X_tr, y_tr)
                    pred = clf.predict(X_te)
                    prob = clf.predict_proba(X_te)[:, 1]

                    rows.append({
                        "variant":        variant,
                        "extractor_type": extractor_type,
                        "extractor":      extractor_id,
                        "cnn_arch":       file_meta.get("cnn_arch", ""),
                        "scaler":         file_meta.get("scaler", ""),
                        "channels":       file_meta.get("channels", ""),
                        "cnn_seed":       file_meta.get("seed", ""),
                        "band":           band,
                        "classifier":     clf_name,
                        "fold":           fold,
                        "bacc":           float(balanced_accuracy_score(y_te, pred)),
                        "auc":            float(roc_auc_score(y_te, prob)),
                        "f1_macro":       float(f1_score(y_te, pred, average="macro")),
                        "n_train":        int(len(y_tr)),
                        "n_test":         int(len(y_te)),
                        "n_g1_train":     int((y_tr == 1).sum()),
                        "n_g2_train":     int((y_tr == 0).sum()),
                    })

    return rows


# ---------------------------------------------------------------------------
# Extractor enumeration
# ---------------------------------------------------------------------------

def enumerate_extractors(project_root: Path) -> list[Path]:
    feature_dir = project_root / "outputs" / "features"
    fs_roi = feature_dir / "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0.npz"
    cnn    = sorted(list_cnn_files(feature_dir))
    return [fs_roi] + cnn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="G1/G2 span classification SLURM array job.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--job-id", type=int)
    grp.add_argument("--list",   action="store_true")
    grp.add_argument("--merge",  action="store_true")

    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).parents[2])
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    project_root = args.project_root
    out_dir      = project_root / "brainage_agg" / "outputs"
    jobs_dir     = out_dir / "span_classif_jobs"
    extractors   = enumerate_extractors(project_root)

    if args.list:
        print(len(extractors))
        for i, p in enumerate(extractors):
            print(f"  {i:4d}  {p.name}")
        return

    if args.merge:
        csvs = sorted(jobs_dir.glob("job_*.csv"))
        if not csvs:
            print("No job CSVs found in", jobs_dir)
            return
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        # Backfill variant for job CSVs produced before this column was added
        if "variant" not in df.columns:
            df["variant"] = "difference"
        else:
            df["variant"] = df["variant"].fillna("difference")
        out_path = out_dir / "span_classif_results.csv"
        df.to_csv(out_path, index=False)
        n_diff  = (df["variant"] == "difference").sum()
        n_concat = (df["variant"] == "concatenation").sum()
        print(f"Merged {len(csvs)} files → {len(df)} rows "
              f"(difference: {n_diff}, concatenation: {n_concat}) → {out_path}")
        return

    job_id = args.job_id
    if not (0 <= job_id < len(extractors)):
        print(f"job-id {job_id} out of range [0, {len(extractors)-1}]", file=sys.stderr)
        sys.exit(1)

    npz_path  = extractors[job_id]
    fs_roi    = extractors[0]
    manifest  = build_manifest(fs_roi, t_mid_seed=T_MID_SEED)

    print(f"[job {job_id}/{len(extractors)-1}] {npz_path.name}")
    t0   = time.time()
    rows = run_span_classification_for_file(npz_path, manifest, n_folds=args.n_folds)
    elapsed = time.time() - t0

    if not rows:
        print("  No results — check cohort/feature alignment.")
        return

    jobs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = jobs_dir / f"job_{job_id:04d}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  {len(rows)} rows → {out_csv}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
