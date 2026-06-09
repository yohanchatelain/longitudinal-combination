"""Convert FreeSurfer ROI group-difference statistics to volumetric brain maps.

Reads a features_fs_roi_*.csv file (columns: feature, t_stat, cohen_d, …) and
paints each ROI's statistic into a MNI152NLin2009cAsym parcellation volume,
producing one NIfTI per measurement type:

  thickness_map_{metric}.nii.gz          cortical thickness (ThickAvg / thickness)
  area_map_{metric}.nii.gz               surface area (SurfArea / area)
  volume_map_{metric}.nii.gz             cortical gray volume (GrayVol / volume)
  subcortical_volume_map_{metric}.nii.gz aseg subcortical structures

Atlas used: tpl-MNI152NLin2009cAsym_res-02_desc-DKT31_dseg.nii.gz
  - 2mm isotropic MNI152NLin2009cAsym space
  - FreeSurfer-compatible label IDs: 1002-2035 cortical (DK), 4-91 subcortical
  - Desikan-Killiany (DK) features (NKI) map directly via FS-LUT.txt
  - Destrieux cortical features (PPMI) are NOT painted (no Destrieux MNI atlas
    in templateflow cache); subcortical features still work for both datasets.

Usage:
    python -m brainage_agg.analysis.roi_to_brain_map \\
        --csv brainage_agg/outputs/group_diff/features_fs_roi_mean.csv \\
        --metric t_stat \\
        --out-dir outputs/brain_maps/nki_mean

    python -m brainage_agg.analysis.roi_to_brain_map \\
        --csv brainage_agg/ppmi_outputs/group_diff/features_fs_roi_mean.csv \\
        --metric cohen_d \\
        --out-dir outputs/brain_maps/ppmi_mean

    # Process all aggregations at once
    python -m brainage_agg.analysis.roi_to_brain_map \\
        --csv-dir brainage_agg/outputs/group_diff \\
        --metric t_stat cohen_d \\
        --out-dir outputs/brain_maps/nki
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Atlas paths
# ---------------------------------------------------------------------------

_TEMPLATEFLOW_DIR = Path.home() / ".cache" / "templateflow" / "tpl-MNI152NLin2009cAsym"

# DKT31 parcellation: FreeSurfer label IDs in MNI152NLin2009cAsym 2mm space.
# Includes cortical DK labels (1002-2035) and subcortical aseg labels (4-91).
_DKT31_ATLAS = _TEMPLATEFLOW_DIR / "tpl-MNI152NLin2009cAsym_res-02_desc-DKT31_dseg.nii.gz"

# MNI T1w brain image used only to supply a clean header/affine if needed.
_MNI_T1W = _TEMPLATEFLOW_DIR / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"


def _find_atlas(override: Path | None = None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Atlas not found: {override}")
        return override
    if _DKT31_ATLAS.exists():
        return _DKT31_ATLAS
    raise FileNotFoundError(
        f"DKT31 atlas not found at {_DKT31_ATLAS}. "
        "Pass --atlas to specify an alternative parcellation NIfTI."
    )


# ---------------------------------------------------------------------------
# LUT parsing
# ---------------------------------------------------------------------------

def load_lut(lut_path: Path) -> dict[str, int]:
    """Return {label_name_lower: label_id} from a FreeSurfer Color LUT file."""
    lut: dict[str, int] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(\S+)")
    for line in lut_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = pattern.match(line)
        if m:
            label_id = int(m.group(1))
            name = m.group(2)
            lut[name.lower()] = label_id
            lut[name] = label_id  # keep original case too
    return lut


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------

# Measurement suffixes used by each dataset
_DK_SUFFIX_MAP = {
    "thickavg": "thickness",
    "surfarea": "area",
    "grayvol": "volume",
    "numvert": "area",       # vertex count ~ area
    "meancurv": None,
    "gauscurv": None,
    "foldind": None,
    "curvind": None,
    "thickstd": None,
}

_PPMI_SUFFIX_MAP = {
    "thickness": "thickness",
    "area": "area",
    "volume": "volume",
}

# Measurement types we output maps for
MEASUREMENT_TYPES = ("thickness", "area", "volume", "subcortical_volume")


def classify_features(df: pd.DataFrame) -> tuple[dict[str, list[str]], str]:
    """Split features by measurement type; return (groups, atlas_type).

    atlas_type is 'dk' (Desikan-Killiany) or 'destrieux'.
    """
    has_cortical = df["feature"].str.match(r"^[lr]h_").any()

    # Detect atlas by region naming convention
    # Destrieux regions start with 'G_', 'S_', 'Lat_', or capital letter after hemi prefix
    sample_cortical = df.loc[df["feature"].str.match(r"^[lr]h_"), "feature"]
    destrieux = False
    if not sample_cortical.empty:
        region_parts = sample_cortical.str.split("_", n=2).str[1]
        destrieux = bool(region_parts.str.match(r"^[GS][_-]").any()
                         or region_parts.str[0].str.isupper().any())

    atlas_type = "destrieux" if destrieux else "dk"

    suffix_map = _PPMI_SUFFIX_MAP if destrieux else _DK_SUFFIX_MAP

    groups: dict[str, list[str]] = {m: [] for m in MEASUREMENT_TYPES}

    for feat in df["feature"]:
        if feat.startswith("lh_") or feat.startswith("rh_"):
            # Cortical feature — identify measurement by last underscore segment
            parts = feat.rsplit("_", 1)
            if len(parts) == 2:
                suffix = parts[1].lower()
                mtype = suffix_map.get(suffix)
                if mtype is not None:
                    groups[mtype].append(feat)
        else:
            # No hemisphere prefix → subcortical (aseg)
            groups["subcortical_volume"].append(feat)

    return groups, atlas_type


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def feature_to_label_id(
    feature: str,
    atlas_type: str,
    lut: dict[str, int],
) -> int | None:
    """Map a feature name like 'lh_bankssts_ThickAvg' or 'lh_G_cuneus_thickness'
    to its FreeSurfer aparc+aseg label ID, or None if not found.

    For subcortical features (no hemi prefix), looks up directly in the LUT
    (e.g. 'Left-Caudate' → 11).
    """
    if not (feature.startswith("lh_") or feature.startswith("rh_")):
        # Subcortical: look up directly; also try stripping '-Proper' suffix
        # (NKI uses 'Left-Thalamus-Proper', LUT has 'Left-Thalamus')
        val = lut.get(feature) or lut.get(feature.lower())
        if val is None and feature.endswith("-Proper"):
            stripped = feature[: -len("-Proper")]
            val = lut.get(stripped) or lut.get(stripped.lower())
        return val

    # Parse: lh_<region>_<suffix>  or  lh_<region>   (for atlas-only names)
    parts = feature.split("_", 1)          # ['lh', rest]
    hemi = parts[0]                        # 'lh' or 'rh'
    rest = parts[1]                        # 'bankssts_ThickAvg' or 'G_cuneus_thickness'

    # Strip measurement suffix (last _ segment for DK; same pattern for Destrieux)
    if atlas_type == "dk":
        # DK: region has no underscores, measurement suffix is last segment
        region_parts = rest.rsplit("_", 1)
        region = region_parts[0] if len(region_parts) == 2 else rest
        # LUT key: ctx-lh-<region>
        lut_key = f"ctx-{hemi}-{region}"
    else:
        # Destrieux: region can contain underscores (e.g., G_and_S_cingul-Ant)
        # measurement suffix is exactly one of: thickness / area / volume
        for meas in ("_thickness", "_area", "_volume"):
            if rest.endswith(meas):
                region = rest[: -len(meas)]
                break
        else:
            region = rest
        # In aparc.a2009s+aseg.mgz (11000-12000 range), LUT uses underscores:
        # ctx_lh_G_cuneus → 11111
        # In aparc+aseg.mgz (1100-1175 range), LUT uses hyphens:
        # ctx-lh-G_cuneus → 1105
        # Prefer the 11000-range (dedicated Destrieux file)
        lut_key_a2009 = f"ctx_{hemi}_{region}"
        lut_key_aparc = f"ctx-{hemi}-{region}"
        val = lut.get(lut_key_a2009) or lut.get(lut_key_a2009.lower())
        if val is not None:
            return val
        lut_key = lut_key_aparc

    return lut.get(lut_key) or lut.get(lut_key.lower())


# ---------------------------------------------------------------------------
# Volume painting
# ---------------------------------------------------------------------------

def build_importance_volume(
    atlas_data: np.ndarray,
    feature_group: list[str],
    df: pd.DataFrame,
    metric: str,
    lut: dict[str, int],
    atlas_type: str,
) -> np.ndarray:
    """Return a float32 volume with each atlas region set to its metric value.

    Background (label 0) remains NaN.
    """
    vol = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    missing = []

    feat_col = df.set_index("feature")[metric]

    for feat in feature_group:
        label_id = feature_to_label_id(feat, atlas_type, lut)
        if label_id is None:
            missing.append(feat)
            continue
        value = feat_col.get(feat)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        mask = atlas_data == label_id
        vol[mask] = float(value)

    if missing:
        print(f"    [warn] {len(missing)} features not found in LUT (first 5: {missing[:5]})")

    return vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(
    csv_path: Path,
    out_dir: Path,
    metrics: list[str],
    atlas_path: Path,
    lut_path: Path,
) -> None:
    print(f"\n{'─'*60}")
    print(f"CSV:     {csv_path}")

    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        print("  [skip] no 'feature' column")
        return

    # Validate requested metrics
    for m in metrics:
        if m not in df.columns:
            print(f"  [warn] metric '{m}' not in CSV columns: {df.columns.tolist()}")
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    feature_groups, atlas_type = classify_features(df)
    print(f"Features: {atlas_type} atlas")
    for mtype, feats in feature_groups.items():
        print(f"  {mtype:25s}: {len(feats):4d} features")
    if atlas_type == "destrieux":
        print(f"  [note] Destrieux cortical features will not be painted "
              f"(no Destrieux MNI atlas in templateflow cache)")

    print(f"Atlas:   {atlas_path}")
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.round(atlas_img.get_fdata()).astype(np.int32)
    affine = atlas_img.affine
    header = atlas_img.header.copy()

    lut = load_lut(lut_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = csv_path.stem  # e.g. features_fs_roi_mean

    for metric in metrics:
        print(f"\n  Metric: {metric}")
        for mtype, feature_group in feature_groups.items():
            if not feature_group:
                continue
            vol = build_importance_volume(
                atlas_data, feature_group, df, metric, lut, atlas_type
            )
            n_painted = int(np.isfinite(vol).sum())
            print(f"    {mtype:25s}: {n_painted:7d} voxels painted")

            out_path = out_dir / f"{mtype}_map_{metric}__{csv_stem}.nii.gz"
            new_header = header.copy()
            new_header.set_data_dtype(np.float32)
            out_img = nib.Nifti1Image(vol, affine=affine, header=new_header)
            nib.save(out_img, str(out_path))
            print(f"    → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Single features_fs_roi_*.csv file")
    src.add_argument(
        "--csv-dir",
        type=Path,
        help="Directory to scan for features_fs_roi_*.csv files",
    )

    parser.add_argument(
        "--metric",
        nargs="+",
        default=["t_stat", "cohen_d"],
        metavar="METRIC",
        help="Column(s) to use as importance value (default: t_stat cohen_d)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/brain_maps"),
        help="Output directory (default: outputs/brain_maps)",
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help=(
            "Parcellation NIfTI with FreeSurfer-compatible label IDs "
            f"(default: {_DKT31_ATLAS})"
        ),
    )
    parser.add_argument(
        "--lut",
        type=Path,
        default=ROOT / "FS-LUT.txt",
        help="FreeSurfer Color LUT file (default: <project-root>/FS-LUT.txt)",
    )

    args = parser.parse_args()

    atlas_path = _find_atlas(args.atlas)
    print(f"Atlas:   {atlas_path}")

    if not args.lut.exists():
        print(f"ERROR: LUT file not found: {args.lut}", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        csv_files = [args.csv]
    else:
        csv_files = sorted(args.csv_dir.glob("features_fs_roi_*.csv"))
        if not csv_files:
            print(f"No features_fs_roi_*.csv files found in {args.csv_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files in {args.csv_dir}")

    for csv_path in csv_files:
        # For --csv-dir mode, mirror the subdir structure under out_dir
        if args.csv_dir:
            rel = csv_path.relative_to(args.csv_dir)
            out_subdir = args.out_dir / rel.parent / rel.stem
        else:
            out_subdir = args.out_dir

        process_csv(
            csv_path=csv_path,
            out_dir=out_subdir,
            metrics=args.metric,
            atlas_path=atlas_path,
            lut_path=args.lut,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
