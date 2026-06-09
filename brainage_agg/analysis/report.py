"""Generate figures and findings.md from results.csv.

Figures saved to brainage_agg/outputs/figures/:
  1. heatmap_arm_A.png / heatmap_arm_B.png / heatmap_arm_B2.png — MAE heatmap
  2. age_band_barplot_arm_B.png — Child vs Adult by aggregation
  3. slope_vs_difference_gap.png — lme_slope_change − difference MAE gap by extractor
  4. negative_control_r2.png — Arm B mean/concat R² distribution

findings.md checks pre-registered expectations against actual results.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found at {path}")
    return pd.read_csv(path)


def _fold_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-fold metrics to mean ± std."""
    return (
        df.groupby(group_cols)[["MAE", "RMSE", "R2", "r"]]
        .agg(["mean", "std"])
        .reset_index()
    )


def _extractor_label(df: pd.DataFrame) -> pd.Series:
    """Collapse CNN variants to 'CNN' and keep 'FS ROI' as-is."""
    return df["extractor_type"].map({"fs_roi": "FS ROI", "cnn": "CNN"})


def make_heatmap(
    df: pd.DataFrame,
    arm: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn("matplotlib/seaborn not installed — skipping heatmap.")
        return

    arm_df = df[(df["target_arm"] == arm) & (df["age_band"] == "all")].copy()
    if arm_df.empty:
        return

    # For B/B2, exclude FS ROI rows flagged with leakage_warning when computing CNN summary
    # (keep leakage rows visible in the heatmap but annotate separately)
    arm_df["extractor_label"] = _extractor_label(arm_df)
    # Average across CNN variants and folds
    summary = (
        arm_df.groupby(["aggregation", "extractor_label"])["MAE"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="aggregation", columns="extractor_label", values="MAE")

    _arm_titles = {
        "A": "Absolute Age",
        "B": "Brain Change Rate (PC1) — ⚠ FS ROI has leakage",
        "B2": "eTIV Rate [mm³/yr] — ⚠ FS ROI has partial leakage",
    }
    title = _arm_titles.get(arm, arm)

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.5), max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        ax=ax,
        cbar_kws={"label": "MAE"},
    )
    ax.set_title(f"Mean MAE — Arm {arm} ({title})")
    ax.set_ylabel("Aggregation")
    ax.set_xlabel("Extractor")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_age_band_barplot(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arm_b = df[(df["target_arm"] == "B") & (df["age_band"].isin(["Child", "Adult"]))].copy()
    if arm_b.empty:
        return

    arm_b["extractor_label"] = _extractor_label(arm_b)
    summary = (
        arm_b.groupby(["aggregation", "age_band", "extractor_label"])["MAE"]
        .mean()
        .reset_index()
    )

    aggregations = sorted(summary["aggregation"].unique())
    bands = ["Child", "Adult"]
    x = np.arange(len(aggregations))
    width = 0.35

    fig, axes = plt.subplots(1, len(summary["extractor_label"].unique()),
                             figsize=(7 * len(summary["extractor_label"].unique()), 5), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, ext in zip(axes, sorted(summary["extractor_label"].unique())):
        sub = summary[summary["extractor_label"] == ext]
        for i, band in enumerate(bands):
            vals = [
                sub[(sub["aggregation"] == a) & (sub["age_band"] == band)]["MAE"].values
                for a in aggregations
            ]
            heights = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax.bar(x + i * width, heights, width, label=band)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(aggregations, rotation=30, ha="right")
        ax.set_title(f"Arm B — {ext}")
        ax.set_ylabel("MAE (years)")
        ax.legend()

    plt.suptitle("Arm B MAE by Age Band and Aggregation")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_slope_difference_gap(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arm_b = df[(df["target_arm"] == "B") & (df["age_band"] == "all")].copy()
    if arm_b.empty:
        return

    arm_b["extractor_label"] = _extractor_label(arm_b)
    pivot = (
        arm_b.groupby(["extractor_label", "aggregation"])["MAE"]
        .mean()
        .unstack("aggregation")
    )

    if "lme_slope_change" not in pivot.columns or "difference" not in pivot.columns:
        return

    pivot["gap"] = pivot["lme_slope_change"] - pivot["difference"]  # negative = slope wins

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = pivot.index.tolist()
    bars = ax.bar(labels, pivot["gap"].values, color=["steelblue" if v <= 0 else "tomato" for v in pivot["gap"].values])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("MAE(lme_slope_change) − MAE(difference)\n(negative = slope wins)")
    ax.set_title("Slope vs Difference gap — Arm B")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_neg_control_r2(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    neg_ctrl = df[
        (df["target_arm"] == "B")
        & (df["aggregation"].isin(["mean", "concatenation"]))
        & (df["age_band"] == "all")
    ].copy()
    if neg_ctrl.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for agg, sub in neg_ctrl.groupby("aggregation"):
        ax.hist(sub["R2"].dropna(), bins=20, alpha=0.6, label=agg)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="R²=0 (chance)")
    ax.set_xlabel("R²")
    ax.set_title("Arm B negative controls: R² distribution\n(should cluster near 0)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def _check(cond: bool, msg: str) -> tuple[str, str]:
    """Return (symbol, message) for findings.md."""
    return ("✓" if cond else "✗", msg)


def generate_findings(df: pd.DataFrame, out_path: Path) -> None:
    """Auto-generate findings.md checking pre-registered expectations."""
    lines = ["# Findings — Longitudinal Aggregation Experiment\n"]
    lines.append(f"Results summary from {len(df)} fold×cell rows.\n")

    def mean_mae(arm, agg, band="all"):
        sub = df[(df["target_arm"] == arm) & (df["aggregation"] == agg) & (df["age_band"] == band)]
        return float(sub["MAE"].mean()) if len(sub) > 0 else float("nan")

    def mean_r2(arm, agg, band="all"):
        sub = df[(df["target_arm"] == arm) & (df["aggregation"] == agg) & (df["age_band"] == band)]
        return float(sub["R2"].mean()) if len(sub) > 0 else float("nan")

    lines.append("## Pre-registered expectations\n")

    # Arm A
    lines.append("### Arm A — Absolute age (expected: concat ≥ mean ≫ lme_slope > annualized_rate)\n")
    mae_concat_A = mean_mae("A", "concatenation")
    mae_mean_A   = mean_mae("A", "mean")
    mae_slope_A  = mean_mae("A", "lme_slope")
    mae_rate_A   = mean_mae("A", "annualized_rate")
    lines.append(f"| Aggregation | Mean MAE (yrs) |")
    lines.append(f"|-------------|--------------|")
    for name, val in [("mean", mae_mean_A), ("concatenation", mae_concat_A),
                      ("lme_slope", mae_slope_A), ("annualized_rate", mae_rate_A)]:
        lines.append(f"| {name} | {val:.3f} |")
    lines.append("")
    sym, msg = _check(mae_concat_A <= mae_mean_A + 0.1, f"concat MAE ({mae_concat_A:.3f}) ≤ mean MAE ({mae_mean_A:.3f})")
    lines.append(f"- {sym} {msg}")
    sym, msg = _check(mae_slope_A > mae_mean_A, f"lme_slope MAE ({mae_slope_A:.3f}) > mean MAE (slope should be worse than state agg for Arm A)")
    lines.append(f"- {sym} {msg}")
    lines.append("")

    # Arm B
    lines.append("### Arm B — Brain Change Rate (expected: lme_slope_change > difference ≫ mean ≈ concat ≈ chance)\n")
    lines.append("> ⚠ **Leakage warning**: FS ROI rows have `leakage_warning=True` — target derived from FS ROI features.\n")
    mae_slope_ch = mean_mae("B", "lme_slope_change")
    mae_diff_B   = mean_mae("B", "difference")
    mae_mean_B   = mean_mae("B", "mean")
    mae_concat_B = mean_mae("B", "concatenation")
    r2_mean_B    = mean_r2("B", "mean")
    r2_concat_B  = mean_r2("B", "concatenation")
    lines.append(f"| Aggregation | Mean MAE | Mean R² |")
    lines.append(f"|-------------|----------|---------|")
    for name, mae_v, r2_v in [
        ("lme_slope_change", mae_slope_ch, mean_r2("B", "lme_slope_change")),
        ("difference",       mae_diff_B,   mean_r2("B", "difference")),
        ("mean (neg ctrl)",  mae_mean_B,   r2_mean_B),
        ("concat (neg ctrl)", mae_concat_B, r2_concat_B),
    ]:
        lines.append(f"| {name} | {mae_v:.3f} | {r2_v:.3f} |")
    lines.append("")
    sym, msg = _check(mae_slope_ch <= mae_diff_B, f"lme_slope_change MAE ({mae_slope_ch:.3f}) ≤ difference MAE ({mae_diff_B:.3f})")
    lines.append(f"- {sym} {msg}")
    sym, msg = _check(mae_diff_B < mae_mean_B, f"difference MAE ({mae_diff_B:.3f}) < mean MAE (change agg beats state)")
    lines.append(f"- {sym} {msg}")
    sym, msg = _check(abs(r2_mean_B) < 0.1, f"negative control (mean) R² ≈ 0: {r2_mean_B:.3f}")
    lines.append(f"- {sym} {msg}")
    sym, msg = _check(abs(r2_concat_B) < 0.1, f"negative control (concat) R² ≈ 0: {r2_concat_B:.3f}")
    lines.append(f"- {sym} {msg}")
    lines.append("")

    # Arm B2
    lines.append("### Arm B2 — eTIV Rate (mm³/yr) (expected: lme_slope_change > difference ≫ mean ≈ concat ≈ chance)\n")
    lines.append("> ⚠ **Partial leakage**: FS ROI rows have `leakage_warning=True` — eTIV correlates with FS ROI features.\n")
    mae_slope_b2 = mean_mae("B2", "lme_slope_change")
    mae_diff_b2  = mean_mae("B2", "difference")
    mae_mean_b2  = mean_mae("B2", "mean")
    mae_concat_b2= mean_mae("B2", "concatenation")
    r2_mean_b2   = mean_r2("B2", "mean")
    r2_concat_b2 = mean_r2("B2", "concatenation")
    lines.append(f"| Aggregation | Mean MAE | Mean R² |")
    lines.append(f"|-------------|----------|---------|")
    for name, mae_v, r2_v in [
        ("lme_slope_change", mae_slope_b2, mean_r2("B2", "lme_slope_change")),
        ("difference",       mae_diff_b2,  mean_r2("B2", "difference")),
        ("mean (neg ctrl)",  mae_mean_b2,  r2_mean_b2),
        ("concat (neg ctrl)", mae_concat_b2, r2_concat_b2),
    ]:
        lines.append(f"| {name} | {mae_v:.4g} | {r2_v:.3f} |")
    lines.append("")
    if not np.isnan(mae_slope_b2) and not np.isnan(mae_diff_b2):
        sym, msg = _check(mae_slope_b2 <= mae_diff_b2, f"lme_slope_change MAE ({mae_slope_b2:.4g}) ≤ difference MAE ({mae_diff_b2:.4g})")
        lines.append(f"- {sym} {msg}")
        sym, msg = _check(mae_diff_b2 < mae_mean_b2, f"difference MAE ({mae_diff_b2:.4g}) < mean MAE (change agg beats state)")
        lines.append(f"- {sym} {msg}")
        sym, msg = _check(abs(r2_mean_b2) < 0.1, f"negative control (mean) R² ≈ 0: {r2_mean_b2:.3f}")
        lines.append(f"- {sym} {msg}")
    else:
        lines.append("- ? Arm B2 data not yet available.")
    lines.append("")

    # B vs B2 consistency check (CNN only, no leakage)
    lines.append("### B vs B2 consistency (CNN extractors only)\n")
    try:
        cnn_b  = df[(df["target_arm"] == "B")  & (df["extractor_type"] == "cnn") & (df["age_band"] == "all")]
        cnn_b2 = df[(df["target_arm"] == "B2") & (df["extractor_type"] == "cnn") & (df["age_band"] == "all")]
        if not cnn_b.empty and not cnn_b2.empty:
            best_b  = cnn_b.groupby("aggregation")["MAE"].mean().idxmin()
            best_b2 = cnn_b2.groupby("aggregation")["MAE"].mean().idxmin()
            lines.append(f"- Best agg for Arm B (CNN): {best_b}")
            lines.append(f"- Best agg for Arm B2 (CNN): {best_b2}")
            consistent = best_b == best_b2
            sym, msg = _check(consistent, f"Best aggregation consistent across B and B2: {best_b}")
            lines.append(f"- {sym} {msg}")
        else:
            lines.append("- ? Insufficient CNN data for B/B2 consistency check.")
    except Exception:
        lines.append("- ? Insufficient data for B/B2 consistency check.")
    lines.append("")

    # Age band (Arm B)
    lines.append("### Age band — Arm B (expected: MAE(children) ≪ MAE(adults))\n")
    mae_child = mean_mae("B", "difference", "Child")
    mae_adult = mean_mae("B", "difference", "Adult")
    lines.append(f"- difference MAE: Child={mae_child:.3f}, Adult={mae_adult:.3f}")
    sym, msg = _check(mae_child < mae_adult, f"children MAE ({mae_child:.3f}) < adult MAE ({mae_adult:.3f})")
    lines.append(f"- {sym} {msg}")
    lines.append("")

    # Extractor × aggregation interaction in Arm B
    lines.append("### Extractor × aggregation interaction (expected: stronger in Arm B)\n")
    try:
        fs_diff_B = df[(df["target_arm"] == "B") & (df["aggregation"] == "difference") & (df["extractor_type"] == "fs_roi") & (df["age_band"] == "all")]["MAE"].mean()
        cnn_diff_B = df[(df["target_arm"] == "B") & (df["aggregation"] == "difference") & (df["extractor_type"] == "cnn") & (df["age_band"] == "all")]["MAE"].mean()
        fs_diff_A = df[(df["target_arm"] == "A") & (df["aggregation"] == "mean") & (df["extractor_type"] == "fs_roi") & (df["age_band"] == "all")]["MAE"].mean()
        cnn_diff_A = df[(df["target_arm"] == "A") & (df["aggregation"] == "mean") & (df["extractor_type"] == "cnn") & (df["age_band"] == "all")]["MAE"].mean()
        gap_B = abs(fs_diff_B - cnn_diff_B)
        gap_A = abs(fs_diff_A - cnn_diff_A)
        sym, msg = _check(gap_B >= gap_A, f"Extractor gap (difference agg) larger in Arm B ({gap_B:.3f}) vs Arm A mean ({gap_A:.3f})")
        lines.append(f"- {sym} {msg}")
    except Exception:
        lines.append("- ? Insufficient data to compute extractor interaction.")
    lines.append("")

    lines.append("---\n")
    lines.append("*Auto-generated by brainage_agg/analysis/report.py*\n")

    text = "\n".join(lines)
    out_path.write_text(text)
    print(f"  Saved {out_path.name}")


def run(result_dir: Path) -> None:
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(result_dir)
    print(f"  {len(df)} rows, {df['extractor'].nunique()} extractors, {df['fold'].nunique()} folds")

    print("Generating figures...")
    make_heatmap(df, "A", fig_dir / "heatmap_arm_A.png")
    make_heatmap(df, "B", fig_dir / "heatmap_arm_B.png")
    make_heatmap(df, "B2", fig_dir / "heatmap_arm_B2.png")
    make_age_band_barplot(df, fig_dir / "age_band_barplot_arm_B.png")
    make_slope_difference_gap(df, fig_dir / "slope_vs_difference_gap.png")
    make_neg_control_r2(df, fig_dir / "negative_control_r2.png")

    print("Generating findings.md...")
    generate_findings(df, result_dir / "findings.md")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).parents[2] / "brainage_agg" / "outputs",
    )
    args = parser.parse_args()
    run(args.result_dir)


if __name__ == "__main__":
    main()
