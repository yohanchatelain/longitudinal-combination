from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "id",
    "participant_id",
    "session",
    "age",
    "sex",
    "timepoint",
    "directory",
]


def load_cohort(
    cohort_csv: str,
    freesurfer_root: str | None = None,
    image_file: str = "mri/brain.mgz",
    mask_file: str = "mri/brainmask.mgz",
    verbose: bool = False,
    use_csv_paths: bool = False,
) -> pd.DataFrame:
    """Load cohort metadata and attach image/mask paths.

    When use_csv_paths=True (or freesurfer_root is None), reads image_path and
    mask_path directly from the CSV instead of constructing them from
    freesurfer_root/directory/image_file. mask_path may be empty — the dataset
    will then fall back to a non-zero-voxel mask derived from the image itself.
    """
    df = pd.read_csv(cohort_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required cohort columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    if use_csv_paths or freesurfer_root is None:
        if "image_path" not in df.columns:
            raise ValueError("CSV must contain image_path when use_csv_paths=True")
        df["image_path"] = df["image_path"].fillna("").astype(str)
        if "mask_path" not in df.columns:
            df["mask_path"] = ""
        else:
            df["mask_path"] = df["mask_path"].fillna("").astype(str)
        exists = df["image_path"].map(lambda p: bool(p) and Path(p).exists())
        dropped = int((~exists).sum())
        if dropped:
            print(f"Warning: dropping {dropped} rows with missing image_path")
            if verbose:
                for _, row in df[~exists].iterrows():
                    print(f"  Missing {row['image_path']}")
        df = df.loc[exists].reset_index(drop=True)
    else:
        root = Path(freesurfer_root)
        df["image_path"] = df["directory"].map(lambda d: str(root / str(d) / image_file))
        df["mask_path"] = df["directory"].map(lambda d: str(root / str(d) / mask_file))

        exists = df["image_path"].map(lambda p: Path(p).exists()) & df["mask_path"].map(
            lambda p: Path(p).exists()
        )

        dropped = int((~exists).sum())
        if dropped:
            print(f"Warning: dropping {dropped} rows with missing brain.mgz/brainmask.mgz")
            if verbose:
                df_with_exists = pd.concat([df, exists.rename("exists")], axis=1)
                for _, row in df_with_exists[~exists].iterrows():
                    print(f"Missing {row['directory']}")
        df = df.loc[exists].reset_index(drop=True)

    if "days_since_enrollment" in df.columns and "days" not in df.columns:
        df["days"] = df["days_since_enrollment"]

    sex_counts = df.groupby("id")["sex"].nunique(dropna=True)
    inconsistent = sex_counts[sex_counts > 1]
    if not inconsistent.empty:
        print(f"Warning: {len(inconsistent)} ids have inconsistent sex labels")

    return df


def load_mgz_float32(path: str) -> np.ndarray:
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def load_mgz_mask(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj) > 0
