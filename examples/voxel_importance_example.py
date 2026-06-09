"""Voxel importance example — end-to-end walkthrough.

Shows how to compute gradient-based voxel importance for a single subject
using the high-level `compute_voxel_importance` API, then save the result
as NIfTI files for inspection in FSLeyes or ITK-SNAP.

Usage
-----
    # With real data:
    python examples/voxel_importance_example.py \
        --brain NKI_FreeSurfer/freesurfer/<subj>/mri/brain.mgz \
        --atlas NKI_FreeSurfer/freesurfer/<subj>/mri/aparc+aseg.mgz \
        --weights outputs/group_diff/features_fs_roi_mean.csv \
        --out /tmp/voxel_importance

    # Synthetic smoke test (no data required):
    python examples/voxel_importance_example.py --synthetic

Run with --help for all options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_synthetic_inputs(n_features: int = 7680):
    """Return (brain, atlas, weights) as numpy arrays for a quick smoke test."""
    rng = np.random.default_rng(42)
    brain  = rng.standard_normal((64, 64, 64)).astype(np.float32)
    # Small atlas: 10 fake regions, label 0 = background
    atlas  = rng.integers(0, 10, size=(64, 64, 64)).astype(np.int32)
    atlas[atlas == 0] = 1  # remove background to keep things simple
    weights = rng.standard_normal(n_features).astype(np.float32)
    return brain, atlas, weights


def _load_mgz(path: str | Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to load MGZ files: pip install nibabel")
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def _load_atlas_mgz(path: str | Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to load MGZ files: pip install nibabel")
    return np.asarray(nib.load(path).get_fdata(), dtype=np.int32)


def _weights_from_csv(csv_path: str | Path, stat_col: str = "t_stat") -> np.ndarray:
    """Load importance weights from a group_diff feature CSV.

    The CSV must have a column named `stat_col` (default: 't_stat').
    Rows are assumed to be in feature-index order.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if stat_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Column '{stat_col}' not found in {csv_path}. Available: {available}"
        )
    return df[stat_col].to_numpy(dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    import torch
    from src.models import CNN3D_DoubleConv, freeze_model
    from brainage_agg.analysis.voxel_attribution import compute_voxel_importance

    # ------------------------------------------------------------------ #
    # 1. Load inputs
    # ------------------------------------------------------------------ #
    if args.synthetic:
        print("Running in synthetic mode — generating random inputs.")
        model     = freeze_model(CNN3D_DoubleConv(in_channels=1))
        brain, atlas, weights = _make_synthetic_inputs(model.out_features)
        lut = {i: f"region_{i}" for i in range(10)}
        ref_affine = np.eye(4)
    else:
        if not args.brain or not args.atlas or not args.weights:
            raise SystemExit("--brain, --atlas, and --weights are required (or use --synthetic).")

        import nibabel as nib
        print(f"Loading brain:  {args.brain}")
        ref_img = nib.load(args.brain)
        brain   = np.asarray(ref_img.get_fdata(), dtype=np.float32)
        ref_affine = ref_img.affine

        print(f"Loading atlas:  {args.atlas}")
        atlas = _load_atlas_mgz(args.atlas)

        print(f"Loading weights: {args.weights}")
        weights = _weights_from_csv(args.weights, stat_col=args.stat_col)

        lut = str(ROOT / "FS-LUT.txt")

        model = freeze_model(CNN3D_DoubleConv(in_channels=1))
        print(f"CNN: {type(model).__name__}, out_features={model.out_features}")

    device = args.device
    print(f"Device: {device}  |  downsample_factor: {args.downsample}")

    # ------------------------------------------------------------------ #
    # 2. Compute voxel importance — one call
    # ------------------------------------------------------------------ #
    result = compute_voxel_importance(
        model=model,
        brain_volume=brain,
        importance_weights=weights,
        atlas=atlas,
        lut=lut,
        device=device,
        downsample_factor=args.downsample,
    )

    # ------------------------------------------------------------------ #
    # 3. Print top regions
    # ------------------------------------------------------------------ #
    print(f"\nTop {args.top_n} regions by absolute importance:")
    print(result.top_regions(n=args.top_n).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 4. Save outputs
    # ------------------------------------------------------------------ #
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        import nibabel as nib

        nib.save(
            nib.Nifti1Image(result.signed_map, ref_affine),
            out_dir / "signed_importance.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(result.abs_map, ref_affine),
            out_dir / "abs_importance.nii.gz",
        )
        csv_path = out_dir / "region_importance.csv"
        result.region_df.to_csv(csv_path, index=False)

        print(f"\nSaved to {out_dir}:")
        print("  signed_importance.nii.gz  — directional gradient map")
        print("  abs_importance.nii.gz     — absolute sensitivity map")
        print(f"  region_importance.csv     — {len(result.region_df)} regions")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--brain",   help="Path to brain.mgz for the subject.")
    p.add_argument("--atlas",   help="Path to aparc+aseg.mgz for the subject.")
    p.add_argument("--weights", help="Path to a group_diff feature CSV (t_stat column).")
    p.add_argument("--stat-col", default="t_stat",
                   help="Column in --weights CSV to use as importance weights (default: t_stat).")
    p.add_argument("--lut",    default=str(ROOT / "FS-LUT.txt"),
                   help="Path to FreeSurfer color LUT (default: FS-LUT.txt in project root).")
    p.add_argument("--out",    help="Output directory for NIfTI maps and CSV.")
    p.add_argument("--device", default="cpu", help="Compute device (default: cpu).")
    p.add_argument("--downsample", type=int, default=2,
                   help="Spatial downsampling factor (default: 2 = 256³→128³).")
    p.add_argument("--top-n",  type=int, default=15,
                   help="Number of top regions to print (default: 15).")
    p.add_argument("--synthetic", action="store_true",
                   help="Run with random synthetic data — no real files needed.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
