"""Plot per-seed attribution maps as a static PNG grid.

Usage:
    python -m brainage_agg.analysis.plot_seed_maps --list
    python -m brainage_agg.analysis.plot_seed_maps --dataset nki --arch cov_pool --mode brainage
    python -m brainage_agg.analysis.plot_seed_maps --dataset ppmi --arch double_conv --mode sex_rf --value signed
    python -m brainage_agg.analysis.plot_seed_maps --out /tmp/seeds.png --threshold 75
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PRIOR_DIR = ROOT / "outputs" / "prior_experiment"

_FS_AFFINE = np.array([
    [-1, 0, 0, 128],
    [0,  0, 1, -128],
    [0, -1, 0, 128],
    [0,  0, 0,   1],
], dtype=float)

_CUT_COORDS = (0, 10, 20)  # axial slices shown per seed row


def _list_modes() -> None:
    for dataset_dir in sorted(PRIOR_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for arch_dir in sorted(dataset_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            for mode_dir in sorted(arch_dir.iterdir()):
                if mode_dir.is_dir() and list(mode_dir.glob("seed-*_abs_map.npy")):
                    n = len(list(mode_dir.glob("seed-*_abs_map.npy")))
                    print(f"  {dataset_dir.name}/{arch_dir.name}/{mode_dir.name}  ({n} seeds)")


def _load_nii(arr: np.ndarray, mask: np.ndarray | None) -> nib.Nifti1Image:
    data = arr.astype(np.float32).copy()
    if mask is not None:
        data[~mask] = 0.0
    return nib.Nifti1Image(data, affine=_FS_AFFINE)


def _bg_nii(dataset: str, arch: str) -> nib.Nifti1Image | None:
    p = PRIOR_DIR / dataset / arch / "mean_brain_minmax_t1.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim == 4:
        arr = arr[0]
    return nib.Nifti1Image(arr.astype(np.float32), affine=_FS_AFFINE)


def _render_row(stat_img: nib.Nifti1Image, bg: nib.Nifti1Image | None,
                label: str, threshold: float, vmax: float, value: str,
                cmap: str, tmp_dir: str) -> str:
    from nilearn import plotting

    bg_path = None
    if bg is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=tmp_dir, delete=False)
        nib.save(bg, tmp.name)
        bg_path = tmp.name

    fig = plt.figure(figsize=(14, 2.2), facecolor="#111")
    try:
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_path if bg_path else False,
            display_mode="z",
            cut_coords=_CUT_COORDS,
            figure=fig,
            black_bg=True,
            cmap=cmap,
            symmetric_cmap=(value == "signed"),
            vmax=vmax,
            vmin=None if value == "signed" else 0.0,
            threshold=threshold,
            colorbar=True,
            title=label,
            annotate=False,
        )
        out = Path(tmp_dir) / f"{label.replace('/', '_').replace(' ', '_')}.png"
        display.savefig(str(out), dpi=90)
        display.close()
    finally:
        plt.close(fig)
        if bg_path:
            try:
                Path(bg_path).unlink(missing_ok=True)
            except Exception:
                pass
    return str(out)


def plot_seed_maps(dataset: str, arch: str, mode: str, value: str = "abs",
                  threshold_pct: int = 70, out: str | None = None) -> Path:
    mode_dir = PRIOR_DIR / dataset / arch / mode
    suffix = "abs_map" if value == "abs" else "signed_map"
    cmap = "hot" if value == "abs" else "RdBu_r"

    seed_files = sorted(
        mode_dir.glob(f"seed-*_{suffix}.npy"),
        key=lambda p: int(re.search(r"seed-(\d+)", p.name).group(1)),
    )
    if not seed_files:
        sys.exit(f"No {suffix} files found in {mode_dir}")

    mask_p = PRIOR_DIR / dataset / arch / "mean_brain_mask_minmax_t1.npy"
    mask = np.load(mask_p) if mask_p.exists() else None
    bg = _bg_nii(dataset, arch)

    # Compute shared vmax + threshold across all seeds
    all_vals: list[float] = []
    arrays: list[np.ndarray] = []
    for fp in seed_files:
        arr = np.load(fp)
        arrays.append(arr)
        nonzero = arr[arr != 0]
        if nonzero.size:
            all_vals.extend(np.abs(nonzero).tolist())

    mean_fp = mode_dir / f"mean_{suffix}.npy"
    mean_arr = np.load(mean_fp) if mean_fp.exists() else None
    if mean_arr is not None:
        nz = mean_arr[mean_arr != 0]
        if nz.size:
            all_vals.extend(np.abs(nz).tolist())

    all_vals_np = np.array(all_vals, dtype=np.float32)
    vmax = float(np.abs(all_vals_np).max()) if all_vals_np.size else 1.0
    threshold = float(np.percentile(np.abs(all_vals_np), threshold_pct)) if threshold_pct > 0 else 1e-9

    rows: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for arr, fp in zip(arrays, seed_files):
            seed_num = re.search(r"seed-(\d+)", fp.name).group(1)
            stat_img = _load_nii(arr, mask)
            label = f"seed-{seed_num}  |  {dataset}/{arch}/{mode}"
            rows.append(_render_row(stat_img, bg, label, threshold, vmax, value, cmap, tmp_dir))

        if mean_arr is not None:
            stat_img = _load_nii(mean_arr, mask)
            rows.append(_render_row(stat_img, bg, f"MEAN  |  {dataset}/{arch}/{mode}",
                                    threshold, vmax, value, cmap, tmp_dir))

        # Stack all rows into one figure
        imgs = [mpimg.imread(r) for r in rows]
        n = len(imgs)
        h, w = imgs[0].shape[:2]
        fig, axes = plt.subplots(n, 1, figsize=(w / 90, n * h / 90 + 0.4),
                                  facecolor="#111", dpi=90)
        if n == 1:
            axes = [axes]
        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02)

        out_path = Path(out) if out else PRIOR_DIR / "figures" / f"seeds_{dataset}_{arch}_{mode}_{value}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=90, bbox_inches="tight", facecolor="#111")
        plt.close(fig)

    print(f"Saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise per-seed attribution maps.")
    parser.add_argument("--list", action="store_true", help="List available dataset/arch/mode combinations")
    parser.add_argument("--dataset", default="nki")
    parser.add_argument("--arch", default="double_conv")
    parser.add_argument("--mode", default="brainage")
    parser.add_argument("--value", choices=["abs", "signed"], default="abs")
    parser.add_argument("--threshold", type=int, default=70,
                        help="Threshold percentile (0–95, default 70)")
    parser.add_argument("--out", default=None, help="Output PNG path (optional)")
    args = parser.parse_args()

    if args.list:
        _list_modes()
        return

    plot_seed_maps(args.dataset, args.arch, args.mode, args.value, args.threshold, args.out)


if __name__ == "__main__":
    main()
