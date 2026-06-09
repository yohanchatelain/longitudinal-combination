from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_studies(s: str) -> set[str]:
    """Parse PostgreSQL-array notation '{A,B,C}' into a Python set of strings."""
    s = str(s).strip().strip("{}")
    return {x.strip() for x in s.split(",")} if s else set()


_DEFAULT_BAND_MAP: dict[str, str] = {
    "Longitudinal_Child": "Child",
    "Longitudinal_Adult": "Adult",
}


def get_band(studies_str: str, band_map: dict[str, str] | None = None) -> str | None:
    """Return band label or None from a studies field string.

    Args:
        studies_str: PostgreSQL-array-encoded studies value, e.g. '{Longitudinal_Child}'
        band_map:    mapping from study name to band label.
                     Defaults to NKI mapping (Child/Adult).
    """
    mapping = band_map if band_map is not None else _DEFAULT_BAND_MAP
    studies = parse_studies(studies_str)
    for study_name, band_label in mapping.items():
        if study_name in studies:
            return band_label
    return None


def build_manifest(
    npz_path: str | Path,
    t_mid_seed: int = 42,
    band_map: dict[str, str] | None = None,
    delta_t_max: float | None = None,
) -> pd.DataFrame:
    """Build a subject-level manifest from a pre-extracted feature .npz file.

    Returns one row per subject with:
      subject_id, n_timepoints, band, delta_t, t_first_idx, t_last_idx,
      t_mid_idx (row index of sampled interior timepoint, NaN if n_tp < 3),
      cohort_all (band set AND n_tp >= 2),
      cohort_concat (band set AND n_tp >= 3),
      exclude_arm_b (delta_t == 0 or cohort_all is False),
      cohort_span1 (cohort_all AND delta_t <= delta_t_max) — only when delta_t_max is set.

    Row indices (t_first_idx etc.) reference rows in the npz arrays.
    t_mid is sampled once with a fixed RNG and must not change across CV folds.
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = np.asarray(data["id"])
    ages = np.asarray(data["age"], dtype=float)
    timepoints = np.asarray(data["timepoint"], dtype=float)
    studies_arr = np.asarray(data["studies"])

    rng = np.random.default_rng(t_mid_seed)

    rows = []
    for subject_id in np.unique(ids):
        mask = ids == subject_id
        row_indices = np.where(mask)[0]
        # Sort timepoints ascending
        tp_order = np.argsort(timepoints[row_indices])
        row_indices = row_indices[tp_order]
        subject_ages = ages[row_indices]
        subject_studies = str(studies_arr[row_indices[0]])

        band = get_band(subject_studies, band_map=band_map)
        n_tp = len(row_indices)
        delta_t = float(subject_ages[-1] - subject_ages[0])

        t_first_idx = int(row_indices[0])
        t_last_idx = int(row_indices[-1])

        # Sample t_mid from interior timepoints (indices [1, -2] inclusive).
        # For n_tp < 3 there are no interior points; set to NaN.
        if n_tp >= 3:
            interior_indices = row_indices[1:-1]
            t_mid_idx = int(rng.choice(interior_indices))
        else:
            t_mid_idx = -1  # sentinel; concat arm excludes these subjects

        cohort_all = band is not None and n_tp >= 2
        cohort_concat = band is not None and n_tp >= 3
        exclude_arm_b = (not cohort_all) or delta_t == 0.0

        row: dict = {
            "subject_id": subject_id,
            "n_timepoints": n_tp,
            "band": band,
            "delta_t": delta_t,
            "ages": list(subject_ages),
            "row_indices": list(row_indices.astype(int)),
            "t_first_idx": t_first_idx,
            "t_last_idx": t_last_idx,
            "t_mid_idx": t_mid_idx,
            "cohort_all": cohort_all,
            "cohort_concat": cohort_concat,
            "exclude_arm_b": exclude_arm_b,
        }
        if delta_t_max is not None:
            row["cohort_span1"] = cohort_all and delta_t <= delta_t_max
        rows.append(row)

    manifest = pd.DataFrame(rows)

    # Log t_mid choices for reproducibility
    n_concat = manifest["cohort_concat"].sum()
    n_all = manifest["cohort_all"].sum()
    n_arm_b = (~manifest["exclude_arm_b"]).sum()
    span1_str = ""
    if "cohort_span1" in manifest.columns:
        n_span1 = manifest["cohort_span1"].sum()
        span1_str = f" | cohort_span1={n_span1} (delta_t<={delta_t_max}yr)"
    print(
        f"Manifest built: {len(manifest)} subjects total | "
        f"cohort_all={n_all} | cohort_concat={n_concat} | "
        f"arm_b_eligible={n_arm_b} (delta_t > 0){span1_str}"
    )

    return manifest


def save_manifest(manifest: pd.DataFrame, out_path: str | Path) -> None:
    """Save manifest to CSV, serialising list columns as JSON strings."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = manifest.copy()
    def _to_json(v):
        if isinstance(v, (list, np.ndarray)):
            return json.dumps([x.item() if hasattr(x, "item") else x for x in v])
        return json.dumps(v)

    for col in ["ages", "row_indices"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_json)
    df.to_csv(out_path, index=False)


_NON_VOL_SUFFIXES = (
    "_ThickAvg", "_ThickStd", "_SurfArea", "_NumVert",
    "_MeanCurv", "_GausCurv", "_FoldInd", "_CurvInd",
)


def _volume_mask(json_path: Path) -> np.ndarray:
    """Return boolean mask (n_features,) — True for volume features (aseg + GrayVol)."""
    names = json.loads(json_path.read_text())["feature_names"]
    return np.array([
        name.endswith("_GrayVol") or not any(name.endswith(s) for s in _NON_VOL_SUFFIXES)
        for name in names
    ])


def compute_brain_change_rate(
    fs_roi_path: str | Path,
    pca_seed: int = 42,
    icv_path: str | Path | None = None,
    band_map: dict[str, str] | None = None,
    delta_t_max: float | None = None,
) -> dict[str, float]:
    """Compute per-subject brain change rate as the PC1 score of annualized FS ROI feature change.

    For each eligible subject (band set, n_tp >= 2, delta_t > 0), computes the
    annualized change vector (f_last - f_first) / delta_t after global z-scoring.
    PCA is fitted on all eligible subjects; the PC1 score is the brain change rate.

    Args:
        fs_roi_path: path to the FS ROI .npz file (scaler-none)
        pca_seed:    random seed for PCA reproducibility
        icv_path:    optional path to the ICV-corrected .npz counterpart.
                     When provided, volume features (aseg + GrayVol) are taken
                     from this file so that eTIV measurement noise does not
                     dominate PC1.  Non-volume features (thickness, area,
                     curvature) are always taken from fs_roi_path unchanged.

    Returns:
        dict mapping subject_id -> brain_change_rate (float) for all eligible subjects
    """
    fs_roi_path = Path(fs_roi_path)
    data = np.load(fs_roi_path, allow_pickle=True)
    ids = np.asarray(data["id"])
    X = np.asarray(data["X"], dtype=float)
    ages = np.asarray(data["age"], dtype=float)
    timepoints = np.asarray(data["timepoint"], dtype=float)
    studies_arr = np.asarray(data["studies"])

    if icv_path is not None:
        json_path = fs_roi_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"Feature names JSON not found at {json_path}. "
                "Re-run extract_roi_features.py to regenerate it."
            )
        vol_mask = _volume_mask(json_path)
        X_icv = np.asarray(np.load(icv_path, allow_pickle=True)["X"], dtype=float)
        X = X.copy()
        X[:, vol_mask] = X_icv[:, vol_mask]
        print(f"  Selective ICV correction: {vol_mask.sum()} volume features corrected, "
              f"{(~vol_mask).sum()} non-volume features kept raw")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X)

    eligible_ids: list[str] = []
    rate_vectors: list[np.ndarray] = []

    for subject_id in np.unique(ids):
        mask = ids == subject_id
        row_indices = np.where(mask)[0]
        tp_order = np.argsort(timepoints[row_indices])
        row_indices = row_indices[tp_order]
        subject_ages = ages[row_indices]
        subject_features = features_scaled[row_indices]
        subject_studies = str(studies_arr[row_indices[0]])

        band = get_band(subject_studies, band_map=band_map)
        if band is None:
            continue
        n_tp = len(row_indices)
        if n_tp < 2:
            continue
        delta_t = float(subject_ages[-1] - subject_ages[0])
        if delta_t <= 0.0:
            continue
        if delta_t_max is not None and delta_t > delta_t_max:
            continue

        rate_vec = (subject_features[-1] - subject_features[0]) / delta_t
        eligible_ids.append(subject_id)
        rate_vectors.append(rate_vec)

    if len(rate_vectors) < 2:
        raise ValueError(
            f"compute_brain_change_rate: only {len(rate_vectors)} eligible subject(s) found. "
            "Need at least 2 to fit PCA."
        )

    rate_matrix = np.array(rate_vectors)  # (n_subjects, n_features)
    pca = PCA(n_components=1, random_state=pca_seed)
    pc_scores = pca.fit_transform(rate_matrix)  # (n_subjects, 1)

    explained = float(pca.explained_variance_ratio_[0])
    print(
        f"Brain change rate: PC1 explains {explained:.1%} of annualized-change variance "
        f"({len(eligible_ids)} eligible subjects)"
    )

    return {sid: float(score[0]) for sid, score in zip(eligible_ids, pc_scores)}


def compute_etiv_rate(
    fs_roi_path: str | Path,
    fs_root: str | Path | None = None,
) -> dict[str, float]:
    """Compute per-subject annualized eTIV change rate.

    Target for Arm B2: (eTIV_last - eTIV_first) / delta_t  [mm³/year].

    Two sources, tried in order:
      1. NPZ 'etiv' array — pre-computed per row (used for PPMI where aseg.stats
         are not available). Present when build_ppmi_roi_npz.py created the file.
      2. FreeSurfer aseg.stats on disk via fs_root — original NKI path.

    WARNING: eTIV is a FreeSurfer-derived measure. When used with FS ROI features
    (which are correlated with eTIV through shared morphometry), there is residual
    leakage. This is less severe than PC1-of-all-ROI but not zero. See Arm B2
    leakage_warning flag in results.

    Args:
        fs_roi_path: path to the FS ROI .npz file
        fs_root:     root directory containing sub-{id}_ses-{session}/ folders
                     (required for aseg.stats path; ignored when NPZ has 'etiv')

    Returns:
        dict mapping subject_id -> etiv_rate (float, mm³/year)
    """
    data = np.load(fs_roi_path, allow_pickle=True)
    ids = np.asarray(data["id"])
    ages = np.asarray(data["age"], dtype=float)
    timepoints = np.asarray(data["timepoint"], dtype=float)

    # Check for pre-computed eTIV array in the NPZ (PPMI path)
    if "etiv" in data:
        etiv_arr = np.asarray(data["etiv"], dtype=float)
        return _compute_etiv_rate_from_array(ids, ages, timepoints, etiv_arr)

    # Fall back to reading aseg.stats from disk (NKI path)
    if fs_root is None:
        raise ValueError(
            "fs_root must be provided when the NPZ does not contain an 'etiv' array."
        )
    directories = np.asarray(data["directory"])
    return _compute_etiv_rate_from_stats(ids, ages, timepoints, directories, Path(fs_root))


def _compute_etiv_rate_from_array(
    ids: np.ndarray,
    ages: np.ndarray,
    timepoints: np.ndarray,
    etiv_arr: np.ndarray,
) -> dict[str, float]:
    """Compute eTIV rate using per-row eTIV values stored in the NPZ."""
    result: dict[str, float] = {}
    n_missing = 0
    for subject_id in np.unique(ids):
        mask = ids == subject_id
        row_indices = np.where(mask)[0]
        tp_order = np.argsort(timepoints[row_indices])
        row_indices = row_indices[tp_order]
        subject_ages = ages[row_indices]
        delta_t = float(subject_ages[-1] - subject_ages[0])
        if delta_t <= 0.0:
            continue
        etiv_first = float(etiv_arr[row_indices[0]])
        etiv_last = float(etiv_arr[row_indices[-1]])
        if not (np.isfinite(etiv_first) and np.isfinite(etiv_last)):
            n_missing += 1
            continue
        result[subject_id] = (etiv_last - etiv_first) / delta_t
    print(
        f"eTIV rate (from NPZ array): computed for {len(result)} subjects "
        f"({n_missing} skipped — NaN eTIV)"
    )
    return result


def _compute_etiv_rate_from_stats(
    ids: np.ndarray,
    ages: np.ndarray,
    timepoints: np.ndarray,
    directories: np.ndarray,
    fs_root: Path,
) -> dict[str, float]:
    """Compute eTIV rate by reading per-session aseg.stats files."""
    result: dict[str, float] = {}
    n_missing = 0
    for subject_id in np.unique(ids):
        mask = ids == subject_id
        row_indices = np.where(mask)[0]
        tp_order = np.argsort(timepoints[row_indices])
        row_indices = row_indices[tp_order]
        subject_ages = ages[row_indices]
        delta_t = float(subject_ages[-1] - subject_ages[0])
        if delta_t <= 0.0:
            continue
        etiv_vals: list[float] = []
        for ri in [row_indices[0], row_indices[-1]]:
            aseg_path = fs_root / str(directories[ri]) / "stats" / "aseg.stats"
            etiv = _parse_etiv(aseg_path)
            if etiv is None:
                break
            etiv_vals.append(etiv)
        if len(etiv_vals) == 2:
            result[subject_id] = (etiv_vals[1] - etiv_vals[0]) / delta_t
        else:
            n_missing += 1
    print(
        f"eTIV rate (from aseg.stats): computed for {len(result)} subjects "
        f"({n_missing} skipped — missing aseg.stats)"
    )
    return result


def _parse_etiv(aseg_path: Path) -> float | None:
    """Extract EstimatedTotalIntraCranialVol from an aseg.stats file."""
    try:
        with open(aseg_path) as f:
            for line in f:
                if "EstimatedTotalIntraCranialVol" in line:
                    return float(line.strip().split(",")[-2].strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    """Load manifest saved by save_manifest, restoring list columns."""
    df = pd.read_csv(csv_path)
    for col in ["ages", "row_indices"]:
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df
