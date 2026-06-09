"""Analysis and visualization for the architectural prior experiment.

Reads outputs from run_prior_experiment.py and produces:
  - Seed agreement heatmaps (Spearman ρ matrix per mode)
  - Top-N region bar charts with std error bars (mean abs attribution)
  - Cross-mode scatter plots (prior vs brainage, prior vs severity, etc.)
  - Uniformity entropy metric per mode
  - NIfTI export of mean abs maps for external brain viewers

Usage:
    python -m brainage_agg.analysis.prior_analysis
    python -m brainage_agg.analysis.prior_analysis \\
        --experiment-dir outputs/prior_experiment \\
        --dataset nki --arch double_conv --top-n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import entropy as scipy_entropy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODES = ["prior", "brainage", "severity"]
MODE_LABELS = {"prior": "Architectural Prior", "brainage": "Brain Age", "severity": "Disease Severity"}
MODE_COLORS = {"prior": "#4878CF", "brainage": "#6ACC65", "severity": "#D65F5F"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mode_dir(exp_dir: Path, dataset: str, arch: str, mode: str) -> Path:
    return exp_dir / dataset / arch / mode


def load_mean_abs(exp_dir: Path, dataset: str, arch: str, mode: str) -> np.ndarray | None:
    p = _mode_dir(exp_dir, dataset, arch, mode) / "mean_abs_map.npy"
    return np.load(p) if p.exists() else None


def load_region_summary(exp_dir: Path, dataset: str, arch: str, mode: str) -> pd.DataFrame | None:
    p = _mode_dir(exp_dir, dataset, arch, mode) / "mean_atlas_regions.csv"
    return pd.read_csv(p) if p.exists() else None


def load_seed_agreement(exp_dir: Path, dataset: str, arch: str, mode: str) -> pd.DataFrame | None:
    p = _mode_dir(exp_dir, dataset, arch, mode) / "seed_agreement.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Uniformity metric
# ---------------------------------------------------------------------------

def voxel_entropy(arr: np.ndarray, n_bins: int = 256) -> float:
    """Entropy of the abs attribution distribution — higher = more uniform."""
    flat = arr.ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return 0.0
    hist, _ = np.histogram(flat, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return float(scipy_entropy(hist))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_seed_agreement(
    exp_dir: Path,
    dataset: str,
    arch: str,
    modes: list[str],
    out_dir: Path,
) -> None:
    avail = [m for m in modes if load_seed_agreement(exp_dir, dataset, arch, m) is not None]
    if not avail:
        return

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, mode in zip(axes[0], avail):
        df = load_seed_agreement(exp_dir, dataset, arch, mode)
        sns.heatmap(
            df, ax=ax, vmin=0, vmax=1, cmap="YlOrRd",
            annot=True, fmt=".2f", square=True,
            cbar_kws={"shrink": 0.8},
        )
        off_diag = df.values[np.triu_indices(len(df), k=1)]
        ax.set_title(f"{MODE_LABELS.get(mode, mode)}\nmean ρ={off_diag.mean():.3f}")
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Seed index")

    fig.suptitle(f"Seed agreement — {dataset.upper()} / {arch}", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"seed_agreement_{dataset}_{arch}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_top_regions(
    exp_dir: Path,
    dataset: str,
    arch: str,
    modes: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> None:
    avail = [m for m in modes if load_region_summary(exp_dir, dataset, arch, m) is not None]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(8 * len(avail), 6), squeeze=False)
    for ax, mode in zip(axes[0], avail):
        df = load_region_summary(exp_dir, dataset, arch, mode).head(top_n)
        y_pos = range(len(df))
        ax.barh(
            y_pos, df["mean_abs"],
            xerr=df.get("std_abs", None),
            color=MODE_COLORS.get(mode, "steelblue"),
            alpha=0.8, capsize=3,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["region"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |attribution|")
        ax.set_title(f"{MODE_LABELS.get(mode, mode)}\nTop {top_n} regions")

    fig.suptitle(f"Region importance — {dataset.upper()} / {arch}", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"top_regions_{dataset}_{arch}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_cross_mode_scatter(
    exp_dir: Path,
    dataset: str,
    arch: str,
    modes: list[str],
    out_dir: Path,
) -> None:
    region_dfs = {
        m: load_region_summary(exp_dir, dataset, arch, m)
        for m in modes
        if load_region_summary(exp_dir, dataset, arch, m) is not None
    }
    if len(region_dfs) < 2:
        return

    mode_list = list(region_dfs)
    pairs = [(m1, m2) for i, m1 in enumerate(mode_list) for m2 in mode_list[i + 1:]]

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    for ax, (m1, m2) in zip(axes[0], pairs):
        df1 = region_dfs[m1].set_index("label_id")["mean_abs"]
        df2 = region_dfs[m2].set_index("label_id")["mean_abs"]
        common = df1.index.intersection(df2.index)
        if len(common) < 5:
            ax.set_visible(False)
            continue
        x, y = df1[common].values, df2[common].values
        ax.scatter(x, y, s=20, alpha=0.6, color="#555555")
        ax.set_xlabel(f"{MODE_LABELS.get(m1, m1)} |attr|")
        ax.set_ylabel(f"{MODE_LABELS.get(m2, m2)} |attr|")

        # Annotate top regions
        top_idx = np.argsort(x + y)[-5:]
        labels = region_dfs[m1].set_index("label_id").reindex(common)["region"].values
        for idx in top_idx:
            ax.annotate(labels[idx], (x[idx], y[idx]), fontsize=6, alpha=0.8)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        ax.set_title(f"{m1} vs {m2}\nSpearman ρ={rho:.3f}  n={len(common)}")

    fig.suptitle(f"Cross-mode region comparison — {dataset.upper()} / {arch}", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"cross_mode_scatter_{dataset}_{arch}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_uniformity_summary(
    exp_dir: Path,
    datasets: list[str],
    archs: list[str],
    modes: list[str],
    out_dir: Path,
) -> None:
    """Bar chart of voxel entropy across (dataset, arch, mode) combos."""
    rows = []
    for dataset in datasets:
        for arch in archs:
            for mode in modes:
                arr = load_mean_abs(exp_dir, dataset, arch, mode)
                if arr is not None:
                    rows.append({
                        "dataset": dataset.upper(),
                        "arch": arch,
                        "mode": MODE_LABELS.get(mode, mode),
                        "entropy": voxel_entropy(arr),
                    })
    if not rows:
        return

    df = pd.DataFrame(rows)
    g = sns.catplot(
        data=df, kind="bar", x="mode", y="entropy",
        hue="arch", col="dataset", height=4, aspect=1.1,
        palette="Set2",
    )
    g.set_axis_labels("Mode", "Voxel entropy (higher = more uniform)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Attribution uniformity across modes", y=1.03, fontsize=13)
    out_path = out_dir / "uniformity_entropy.png"
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  Saved {out_path.name}")

    df.to_csv(out_dir / "uniformity_entropy.csv", index=False)


# ---------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------

def export_nifti(
    exp_dir: Path,
    dataset: str,
    arch: str,
    modes: list[str],
    reference_mgz: str | None,
) -> None:
    """Save mean abs maps as NIfTI files for external viewers."""
    affine = np.eye(4)
    if reference_mgz:
        try:
            affine = nib.load(reference_mgz).affine
        except Exception:
            pass

    for mode in modes:
        arr = load_mean_abs(exp_dir, dataset, arch, mode)
        if arr is None:
            continue
        nii = nib.Nifti1Image(arr, affine)
        out_path = exp_dir / dataset / arch / mode / "mean_abs_map.nii.gz"
        nib.save(nii, str(out_path))
        print(f"  NIfTI saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    exp_dir: Path,
    datasets: list[str],
    archs: list[str],
    modes: list[str],
    top_n: int,
    export_nii: bool,
) -> None:
    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for arch in archs:
            print(f"\n=== {dataset.upper()} / {arch} ===")
            plot_seed_agreement(exp_dir, dataset, arch, modes, out_dir)
            plot_top_regions(exp_dir, dataset, arch, modes, out_dir, top_n=top_n)
            plot_cross_mode_scatter(exp_dir, dataset, arch, modes, out_dir)
            if export_nii:
                # Try to find a reference image for affine from config
                cfg_path = ROOT / "brainage_agg" / (
                    "ppmi_config.yaml" if dataset == "ppmi" else "config.yaml"
                )
                reference_mgz = None
                try:
                    with open(cfg_path) as fh:
                        cfg = yaml.safe_load(fh)
                    from src.io import load_cohort
                    cohort_csv = ROOT / cfg["data"]["cohort_csv"]
                    fs_root = cfg["data"].get("fs_root")
                    if fs_root:
                        cohort = load_cohort(str(cohort_csv), freesurfer_root=str(ROOT / fs_root))
                        if not cohort.empty:
                            reference_mgz = cohort.iloc[0]["image_path"]
                except Exception:
                    pass
                export_nifti(exp_dir, dataset, arch, modes, reference_mgz)

    plot_uniformity_summary(exp_dir, datasets, archs, modes, out_dir)
    print(f"\nAll figures saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize architectural prior experiment outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-dir", type=Path,
                        default=ROOT / "outputs" / "prior_experiment",
                        metavar="DIR")
    parser.add_argument("--dataset", nargs="+", default=["nki", "ppmi"],
                        choices=["nki", "ppmi"], dest="datasets", metavar="DATASET")
    parser.add_argument("--arch", nargs="+", default=["double_conv", "cov_pool"],
                        choices=["double_conv", "cov_pool"], dest="archs", metavar="ARCH")
    parser.add_argument("--modes", nargs="+", default=MODES,
                        choices=MODES, metavar="MODE")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top regions to show in bar charts.")
    parser.add_argument("--export-nifti", action="store_true",
                        help="Export mean abs maps as .nii.gz files.")
    args = parser.parse_args()

    run(
        exp_dir=args.experiment_dir,
        datasets=args.datasets,
        archs=args.archs,
        modes=args.modes,
        top_n=args.top_n,
        export_nii=args.export_nifti,
    )


if __name__ == "__main__":
    main()
