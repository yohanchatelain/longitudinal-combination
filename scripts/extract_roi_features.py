#!/usr/bin/env python
"""Extract FreeSurfer ROI features from aseg.stats and aparc.stats files.

Outputs .npz files in the same format as extract_all_features.py so that
run_all_evaluations.py can process them without modification.

Feature sets produced
---------------------
aseg_vol      : Volume_mm3 per subcortical structure from aseg.stats
aparc_thick   : ThickAvg per region, lh + rh (Desikan atlas)
aparc_vol     : GrayVol per region, lh + rh
aparc_area    : SurfArea per region, lh + rh
aseg_aparc    : concatenation of aseg_vol + aparc_thick + aparc_vol + aparc_area

Output files
------------
outputs/features/features__model-freesurfer_roi__scaler-<scaler>__channels-<roi_set>__seed-0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_cohort
from src.utils import ensure_dir, load_config, write_json

CNN_ARCH = "freesurfer_roi"
SEED = 0

ASEG_COLS = ["NVoxels", "Volume_mm3", "NormMean", "NormStdDev", "Min", "Max", "Range"]
APARC_COLS = ["NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd",
              "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]

METADATA_KEYS = [
    "id", "participant_id", "session", "session_type", "age", "sex",
    "timepoint", "directory", "days", "days_since_enrollment",
    "studies", "study_num", "image_path", "mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--scalers", nargs="+", default=["none", "icv"],
        choices=["none", "icv"],
        help="'none' = raw values; 'icv' = volumes divided by eTIV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stats file parsers
# ---------------------------------------------------------------------------

def _read_stats_table(path: Path) -> pd.DataFrame:
    """Parse the data table from a FreeSurfer .stats file."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        rows.append(line.split())
    return rows


def parse_aseg(path: Path) -> tuple[pd.Series, float]:
    """Return (volume Series indexed by StructName, eTIV).

    Volume values are in mm³.
    """
    rows = _read_stats_table(path)
    # columns: Index SegId NVoxels Volume_mm3 StructName NormMean NormStdDev Min Max Range
    records = {}
    for r in rows:
        if len(r) < 10:
            continue
        struct = r[4]
        try:
            vol = float(r[3])
        except ValueError:
            continue
        records[struct] = vol

    etiv = np.nan
    for line in path.read_text().splitlines():
        if "EstimatedTotalIntraCranialVol" in line and "Measure" in line:
            try:
                etiv = float(line.strip().split(",")[-2].strip())
            except (ValueError, IndexError):
                pass

    return pd.Series(records, dtype=np.float32), etiv


def parse_aparc(path: Path) -> pd.DataFrame:
    """Return DataFrame with one row per region, columns = APARC_COLS."""
    rows = _read_stats_table(path)
    # columns: StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    records = []
    for r in rows:
        if len(r) < 10:
            continue
        rec = {"StructName": r[0]}
        for i, col in enumerate(APARC_COLS, start=1):
            try:
                rec[col] = float(r[i])
            except (ValueError, IndexError):
                rec[col] = np.nan
        records.append(rec)
    return pd.DataFrame(records).set_index("StructName")


# ---------------------------------------------------------------------------
# Feature assembly per subject
# ---------------------------------------------------------------------------

def subject_roi_features(
    directory: str,
    fs_root: Path,
) -> tuple[pd.Series | None, pd.DataFrame | None, pd.DataFrame | None, float]:
    """Return (aseg_vol, lh_aparc, rh_aparc, etiv) for one subject.

    Returns (None, None, None, nan) if any required file is missing.
    """
    subj_dir = fs_root / directory / "stats"
    aseg_path = subj_dir / "aseg.stats"
    lh_path = subj_dir / "lh.aparc.stats"
    rh_path = subj_dir / "rh.aparc.stats"

    if not (aseg_path.exists() and lh_path.exists() and rh_path.exists()):
        return None, None, None, np.nan

    try:
        aseg_vol, etiv = parse_aseg(aseg_path)
        lh = parse_aparc(lh_path)
        rh = parse_aparc(rh_path)
    except Exception:
        return None, None, None, np.nan

    return aseg_vol, lh, rh, etiv


# ---------------------------------------------------------------------------
# Build feature matrices
# ---------------------------------------------------------------------------

def build_feature_matrices(df: pd.DataFrame, fs_root: Path, verbose: bool) -> dict:
    """Parse stats files for all rows and return feature DataFrames per roi_set."""
    aseg_rows, lh_rows, rh_rows, etiv_vals, valid_mask = [], [], [], [], []

    for _, row in df.iterrows():
        aseg_vol, lh, rh, etiv = subject_roi_features(row["directory"], fs_root)
        if aseg_vol is None:
            if verbose:
                print(f"  Missing stats: {row['directory']}")
            valid_mask.append(False)
            aseg_rows.append(None)
            lh_rows.append(None)
            rh_rows.append(None)
            etiv_vals.append(np.nan)
        else:
            valid_mask.append(True)
            aseg_rows.append(aseg_vol)
            lh_rows.append(lh)
            rh_rows.append(rh)
            etiv_vals.append(etiv)

    valid = np.array(valid_mask)
    df_valid = df[valid].reset_index(drop=True)
    etiv_arr = np.array([v for v, m in zip(etiv_vals, valid_mask) if m], dtype=np.float32)

    aseg_df = pd.DataFrame([r for r, m in zip(aseg_rows, valid_mask) if m]).astype(np.float32)

    # Build aparc matrices with consistent column order
    def _aparc_flat(rows_list, col):
        hemis = []
        for hemi_label, rows in [("lh", [r for r, m in zip(lh_rows, valid_mask) if m]),
                                  ("rh", [r for r, m in zip(rh_rows, valid_mask) if m])]:
            # align to common region index from first valid row
            all_df = pd.DataFrame({
                f"{hemi_label}_{row.index[k]}_{col}": [r[col].iloc[k] if col in r.columns and k < len(r) else np.nan
                                                        for r in rows]
                for k, _ in enumerate(rows[0].index)
            } if rows else {}).astype(np.float32)
            hemis.append(all_df)
        return pd.concat(hemis, axis=1)

    # Simpler: stack each hemi directly
    def _aparc_matrix(col):
        lh_mat = pd.DataFrame(
            [r[col].rename(lambda n: f"lh_{n}") if col in r.columns else pd.Series(dtype=np.float32)
             for r, m in zip(lh_rows, valid_mask) if m],
            dtype=np.float32,
        )
        rh_mat = pd.DataFrame(
            [r[col].rename(lambda n: f"rh_{n}") if col in r.columns else pd.Series(dtype=np.float32)
             for r, m in zip(rh_rows, valid_mask) if m],
            dtype=np.float32,
        )
        return pd.concat([lh_mat.reset_index(drop=True), rh_mat.reset_index(drop=True)], axis=1)

    aparc_thick = _aparc_matrix("ThickAvg")
    aparc_vol   = _aparc_matrix("GrayVol")
    aparc_area  = _aparc_matrix("SurfArea")

    # all_roi: every aparc column for lh+rh, plus aseg volumes
    def _aparc_all_cols():
        parts = []
        for hemi_label, rows in [("lh", [r for r, m in zip(lh_rows, valid_mask) if m]),
                                  ("rh", [r for r, m in zip(rh_rows, valid_mask) if m])]:
            hemi_df = pd.DataFrame(
                [{f"{hemi_label}_{region}_{col}": row.loc[region, col]
                  for region in row.index
                  for col in row.columns}
                 for row in rows],
                dtype=np.float32,
            )
            parts.append(hemi_df.reset_index(drop=True))
        return pd.concat(parts, axis=1)

    aparc_all = _aparc_all_cols()

    n_missing_aseg = valid.sum() - aseg_df.notna().all(axis=1).sum()
    print(f"  Valid subjects: {valid.sum()} / {len(df)}  "
          f"(missing stats: {(~valid).sum()}, aseg NaN rows: {n_missing_aseg})")

    aseg_X       = aseg_df.to_numpy(dtype=np.float32)
    aparc_X      = aparc_all.to_numpy(dtype=np.float32)
    all_roi_X    = np.concatenate([aseg_X, aparc_X], axis=1)
    all_roi_names = list(aseg_df.columns) + list(aparc_all.columns)

    feature_sets = {
        "aseg_vol":    aseg_X,
        "aparc_thick": aparc_thick.to_numpy(dtype=np.float32),
        "aparc_vol":   aparc_vol.to_numpy(dtype=np.float32),
        "aparc_area":  aparc_area.to_numpy(dtype=np.float32),
        "aseg_aparc":  np.concatenate([
            aseg_X,
            aparc_thick.to_numpy(dtype=np.float32),
            aparc_vol.to_numpy(dtype=np.float32),
            aparc_area.to_numpy(dtype=np.float32),
        ], axis=1),
        "all_roi":     all_roi_X,
    }
    feature_names = {
        "aseg_vol":    list(aseg_df.columns),
        "aparc_thick": list(aparc_thick.columns),
        "aparc_vol":   list(aparc_vol.columns),
        "aparc_area":  list(aparc_area.columns),
        "aseg_aparc":  list(aseg_df.columns) + list(aparc_thick.columns) + list(aparc_vol.columns) + list(aparc_area.columns),
        "all_roi":     all_roi_names,
    }

    return {
        "feature_sets": feature_sets,
        "feature_names": feature_names,
        "df_valid": df_valid,
        "etiv": etiv_arr,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# ICV correction
# ---------------------------------------------------------------------------

def apply_icv(X: np.ndarray, etiv: np.ndarray) -> np.ndarray:
    """Divide each feature by eTIV (per subject). NaN where eTIV is missing."""
    out = X.copy()
    valid_etiv = np.isfinite(etiv) & (etiv > 0)
    out[valid_etiv] = out[valid_etiv] / etiv[valid_etiv, None]
    out[~valid_etiv] = np.nan
    return out


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def metadata_arrays(df: pd.DataFrame) -> dict:
    return {key: df[key].values for key in METADATA_KEYS if key in df.columns}


def output_paths(out_dir: Path, scaler: str, roi_set: str) -> tuple[Path, Path]:
    stem = f"features__model-{CNN_ARCH}__scaler-{scaler}__channels-{roi_set}__seed-{SEED}"
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def save(out_path, json_path, X, df, scaler, roi_set, feature_names):
    np.savez_compressed(
        out_path,
        X=X,
        **metadata_arrays(df),
        scaler=scaler,
        channels=roi_set,
        seed=SEED,
        cnn_arch=CNN_ARCH,
    )
    write_json(json_path, {
        "cnn_arch": CNN_ARCH,
        "scaler": scaler,
        "channels": roi_set,
        "seed": SEED,
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "n_nan_rows": int(np.isnan(X).any(axis=1).sum()),
        "feature_names": feature_names,
    })
    print(f"  Saved {out_path.name}  shape={X.shape}  nan_rows={np.isnan(X).any(axis=1).sum()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    out_dir = ensure_dir(config["output"]["feature_dir"])
    fs_root = Path(data_cfg["freesurfer_root"])

    df = load_cohort(
        data_cfg["cohort_csv"],
        data_cfg["freesurfer_root"],
        image_file=data_cfg.get("image_file", "mri/brain.mgz"),
        mask_file=data_cfg.get("mask_file", "mri/brainmask.mgz"),
        verbose=args.verbose,
    )
    print(f"Cohort loaded: {len(df)} rows")

    print("Parsing FreeSurfer stats files...")
    result = build_feature_matrices(df, fs_root, args.verbose)
    feature_sets = result["feature_sets"]
    feature_names = result["feature_names"]
    df_valid = result["df_valid"]
    etiv = result["etiv"]

    volume_sets = {"aseg_vol", "aparc_vol", "aseg_aparc", "all_roi"}

    for roi_set, X_raw in feature_sets.items():
        for scaler in args.scalers:
            out_path, json_path = output_paths(out_dir, scaler, roi_set)
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing {out_path.name}")
                continue

            if scaler == "icv" and roi_set not in volume_sets:
                # ICV correction only makes sense for volume features
                if args.verbose:
                    print(f"  Skipping ICV scaler for non-volume set '{roi_set}'")
                continue

            X = apply_icv(X_raw, etiv) if scaler == "icv" else X_raw
            save(out_path, json_path, X, df_valid, scaler, roi_set, feature_names[roi_set])

    print("Done.")


if __name__ == "__main__":
    main()
