#!/usr/bin/env python3
"""Build PPMI FreeSurfer ROI feature NPZ from living-park subject-level CSVs.

Instead of parsing raw aseg/aparc.stats files (not fully available for PPMI),
this script pivots the pre-computed per-subject CSV files into a single NPZ
matching the NKI output format expected by brainage_agg.

Feature sources (living-park/csv/):
  subject-thickness-mean.csv   → cortical thickness per ROI (Destrieux atlas)
  subject-area-mean.csv        → cortical surface area per ROI
  subject-volume-mean.csv      → cortical gray matter volume per ROI
  subject-subcortical-volume-mean.csv → subcortical volumes (incl. eTIV)

Output NPZ arrays:
  X           (n_rows, n_features)  — feature matrix (raw, unscaled)
  id          (n_rows,)             — subject PATNO string
  participant_id, session, session_type, age, sex, timepoint, directory,
  days, days_since_enrollment, studies, study_num, image_path, mask_path
  etiv        (n_rows,)             — eTIV in mm³ (for compute_etiv_rate)
  scaler, channels, seed, cnn_arch  — scalars (metadata)

Also writes a JSON sidecar with feature_names.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIVING_PARK = Path("/mnt/lustre/ychatel/living-park")
CSV_DIR = LIVING_PARK / "csv"

FEATURE_CSVS = {
    "thickness": CSV_DIR / "subject-thickness-mean.csv",
    "area": CSV_DIR / "subject-area-mean.csv",
    "volume": CSV_DIR / "subject-volume-mean.csv",
    "subcortical": CSV_DIR / "subject-subcortical-volume-mean.csv",
}

# Global / non-ROI measures to exclude from cortical feature columns
# (they appear in the thickness/area CSVs but are not cortical ROI features)
GLOBAL_EXCLUDES = {"BrainSegVolNotVent", "eTIV", "BrainSegVol-to-eTIV", "MaskVol-to-eTIV"}

# Subcortical ROIs to keep as segmentation features (exclude global brain measures)
SUBCORTICAL_EXCLUDES = {
    "BrainSegVol", "BrainSegVolNotVent", "BrainSegVol-to-eTIV", "MaskVol-to-eTIV",
    "eTIV", "CerebralWhiteMatterVol", "CortexVol",
}
ETIV_ROI = "EstimatedTotalIntraCranialVol"


def load_wide(csv_path: Path, feature_type: str) -> pd.DataFrame:
    """Load a long-format CSV and pivot to wide: subjid × feature_name."""
    df = pd.read_csv(csv_path)
    if "hemi" in df.columns:
        # Cortical: feature_name = {hemi}_{ROI}
        df = df[~df["ROI"].isin(GLOBAL_EXCLUDES)].copy()
        df["feature_name"] = df["hemi"] + "_" + df["ROI"]
    else:
        # Subcortical: no hemi prefix; exclude global measures (keep actual ROIs)
        df = df[~df["ROI"].isin(SUBCORTICAL_EXCLUDES | {ETIV_ROI})].copy()
        df["feature_name"] = df["ROI"]
    wide = df.pivot_table(index="subjid", columns="feature_name", values="measure", aggfunc="first")
    wide.columns.name = None
    wide = wide.reset_index()
    return wide


def load_etiv(csv_path: Path) -> pd.Series:
    """Return Series: subjid → eTIV (mm³)."""
    df = pd.read_csv(csv_path)
    etiv_rows = df[df["ROI"] == ETIV_ROI][["subjid", "measure"]].drop_duplicates("subjid")
    return etiv_rows.set_index("subjid")["measure"]


def build_npz(cohort_csv: Path, output_dir: Path, dry_run: bool = False) -> None:
    cohort = pd.read_csv(cohort_csv)
    if dry_run:
        cohort = cohort[cohort["id"].isin(cohort["id"].unique()[:10])].copy()
        print(f"[DRY RUN] Using {cohort['id'].nunique()} subjects")

    # Build subjid key matching the CSV format: sub-{PATNO}_ses-{SESSION}
    cohort["subjid"] = cohort["participant_id"] + "_ses-" + cohort["session"]

    # --- Load feature tables ---
    print("Loading feature CSVs...")
    wide = {}
    for name, path in FEATURE_CSVS.items():
        wide[name] = load_wide(path, name)
        print(f"  {name}: {wide[name].shape[1]-1} features, {len(wide[name])} subjects")

    etiv_series = load_etiv(FEATURE_CSVS["subcortical"])

    # --- Merge all feature tables ---
    combined = wide["thickness"]
    for name in ["area", "volume", "subcortical"]:
        combined = combined.merge(wide[name], on="subjid", how="outer")
    print(f"Combined feature table: {len(combined)} rows, {combined.shape[1]-1} features")

    # --- Join with cohort ---
    df = cohort.merge(combined, on="subjid", how="inner")
    print(f"After cohort join: {len(df)} rows ({df['id'].nunique()} subjects)")

    # --- Feature matrix ---
    feature_cols = [c for c in combined.columns if c != "subjid"]
    X = df[feature_cols].values.astype(np.float32)
    n_nan = np.isnan(X).any(axis=1).sum()
    print(f"Feature matrix shape: {X.shape}, rows with NaN: {n_nan}")

    # --- eTIV array ---
    etiv_vals = df["subjid"].map(etiv_series).values.astype(np.float32)

    # --- Metadata arrays ---
    meta_keys = [
        "id", "participant_id", "session", "session_type", "age", "sex",
        "timepoint", "directory", "days", "days_since_enrollment",
        "studies", "study_num", "image_path", "mask_path",
    ]
    meta = {k: df[k].values for k in meta_keys if k in df.columns}

    # --- Save NPZ ---
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "features__model-freesurfer_roi__scaler-none__channels-all_roi__seed-0"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        X=X,
        etiv=etiv_vals,
        scaler="none",
        channels="all_roi",
        seed=0,
        cnn_arch="freesurfer_roi",
        **meta,
    )
    with open(json_path, "w") as f:
        json.dump({
            "cnn_arch": "freesurfer_roi",
            "scaler": "none",
            "channels": "all_roi",
            "seed": 0,
            "n_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "n_nan_rows": int(n_nan),
            "feature_names": feature_cols,
        }, f, indent=2)

    print(f"\nSaved: {npz_path}  shape={X.shape}")
    print(f"JSON:  {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PPMI FS ROI feature NPZ.")
    parser.add_argument(
        "--cohort", type=Path,
        default=Path(__file__).parents[1] / "PPMI_data" / "cohort_longitudinal.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parents[1] / "PPMI_data" / "features",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_npz(args.cohort, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
