"""Factorial experiment: iterate extractor × arm × aggregation × band.

Writes:
  <result_dir>/results.csv          — long-format per-fold metrics
  <result_dir>/manifest.csv         — subject manifest
  <result_dir>/seeds.json           — all seeds used
  <result_dir>/config_snapshot.yaml — config at run time

Flags:
  --config PATH   config file (default: project_root/brainage_agg/config.yaml)
  --arms A B B2   subset of arms to evaluate (default: all)
  --append        append new rows to existing results.csv rather than overwriting
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Allow running as script from project root
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from brainage_agg.agg.aggregations import AGGREGATION_SPECS
from brainage_agg.data.manifest import build_manifest, save_manifest, compute_brain_change_rate, compute_etiv_rate
from brainage_agg.features.loader import load_features, align_to_manifest, list_cnn_files
from brainage_agg.modeling.train import evaluate_cell


ARM_A_AGGREGATIONS = ["mean", "concatenation", "annualized_rate", "lme_slope"]
ARM_B_AGGREGATIONS = ["annualized_rate", "difference", "lme_slope_change", "mean", "concatenation"]
ARM_B2_AGGREGATIONS = ["annualized_rate", "difference", "lme_slope_change", "mean", "concatenation"]
ALL_ARMS = [("A", ARM_A_AGGREGATIONS), ("B", ARM_B_AGGREGATIONS), ("B2", ARM_B2_AGGREGATIONS)]


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    """Make all relative paths absolute from project root."""
    cfg["data"]["cohort_csv"] = str(project_root / cfg["data"]["cohort_csv"])
    cfg["data"]["feature_dir"] = str(project_root / cfg["data"]["feature_dir"])
    cfg["output"]["result_dir"] = str(project_root / cfg["output"]["result_dir"])
    # fs_root may be absolute already; only prepend project_root if relative
    fs_root = cfg["data"].get("fs_root", "")
    if fs_root and not Path(fs_root).is_absolute():
        cfg["data"]["fs_root"] = str(project_root / fs_root)
    return cfg


def enumerate_extractors(cfg: dict) -> list[Path]:
    """Return list of all feature .npz files to evaluate."""
    feature_dir = Path(cfg["data"]["feature_dir"])
    fs_roi_path = feature_dir / cfg["data"]["fs_roi_file"]
    cnn_paths = list_cnn_files(feature_dir)
    return [fs_roi_path] + cnn_paths


def run(
    project_root: Path,
    config_path: Path | None = None,
    dry_run: bool = False,
    n_dry_subjects: int = 30,
    skip_lme: bool = False,
    arms: list[str] | None = None,
    append: bool = False,
) -> None:
    if config_path is None:
        config_path = project_root / "brainage_agg" / "config.yaml"
    cfg = resolve_paths(load_config(config_path), project_root)

    out_dir = Path(cfg["output"]["result_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy(config_path, out_dir / "config_snapshot.yaml")

    # Save seeds
    seeds = {
        "t_mid": cfg["seeds"]["t_mid"],
        "cv_outer": cfg["seeds"]["cv_outer"],
    }
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=2))

    # --- Derive bands from band_map (default: NKI Child/Adult) ---
    band_map: dict[str, str] | None = cfg["data"].get("band_map")
    if band_map:
        bands = ["all"] + sorted(set(band_map.values()))
    else:
        bands = ["all", "Child", "Adult"]

    # --- Build manifest from FS ROI file (same row ordering for all files) ---
    fs_roi_path = Path(cfg["data"]["feature_dir"]) / cfg["data"]["fs_roi_file"]
    delta_t_max: float | None = cfg["data"].get("delta_t_max")
    print("Building manifest...")
    manifest = build_manifest(
        fs_roi_path,
        t_mid_seed=cfg["seeds"]["t_mid"],
        band_map=band_map,
        delta_t_max=delta_t_max,
    )

    # --- Compute Arm B target: brain change rate (PC1 of annualized FS ROI change) ---
    # Uses the ICV-corrected feature file so that eTIV measurement noise (which
    # propagates identically into all raw volumes) does not dominate PC1.
    # Fitted on all eligible subjects before any dry-run subsetting.
    print("Computing brain change rate (Arm B target)...")
    targets_cfg = cfg.get("targets", {})
    pca_seed = targets_cfg.get("brain_change_rate_pca_seed", 42)
    icv_file = targets_cfg.get("brain_change_rate_icv_file")
    icv_path = (Path(cfg["data"]["feature_dir"]) / icv_file) if icv_file else None
    brain_change_rates = compute_brain_change_rate(
        fs_roi_path,
        pca_seed=pca_seed,
        icv_path=icv_path,
        band_map=band_map,
        delta_t_max=delta_t_max,
    )
    manifest["brain_change_rate"] = manifest["subject_id"].map(brain_change_rates)
    n_bcr = manifest["brain_change_rate"].notna().sum()
    print(f"  brain_change_rate set for {n_bcr}/{len(manifest)} subjects")

    # --- Compute Arm B2 target: annualized eTIV rate ---
    # (eTIV_last - eTIV_first) / delta_t  [mm³/year]
    # WARNING: eTIV is a FreeSurfer measure correlated with FS ROI features.
    # FS ROI results in Arm B2 carry leakage_warning=True in the output CSV.
    print("Computing eTIV rate (Arm B2 target)...")
    fs_root_str = cfg["data"].get("fs_root")
    fs_root = Path(fs_root_str) if fs_root_str else None
    etiv_rates = compute_etiv_rate(fs_roi_path, fs_root=fs_root)
    manifest["etiv_rate"] = manifest["subject_id"].map(etiv_rates)
    n_etiv = manifest["etiv_rate"].notna().sum()
    print(f"  etiv_rate set for {n_etiv}/{len(manifest)} subjects")

    if dry_run:
        # Restrict to a small subset of subjects for fast testing
        keep = manifest[manifest["cohort_all"]]["subject_id"].iloc[:n_dry_subjects].tolist()
        manifest = manifest[manifest["subject_id"].isin(keep)].reset_index(drop=True)
        print(f"[DRY RUN] Using {len(manifest)} subjects.")

    save_manifest(manifest, out_dir / "manifest.csv")

    cv_config = {
        "n_outer_folds": cfg["cv"]["n_outer_folds"],
        "n_inner_folds": cfg["cv"]["n_inner_folds"],
        "ridge_alphas": cfg["cv"]["ridge_alphas"],
        "lme_threshold": cfg["lme"]["use_statsmodels_threshold"],
    }

    extractors = enumerate_extractors(cfg)
    if dry_run:
        extractors = extractors[:3]  # FS ROI + 2 CNN files

    # Build factorial job list (filter by --arms if provided)
    active_arms = {a.upper() for a in arms} if arms else {arm for arm, _ in ALL_ARMS}
    jobs = []
    for npz_path in extractors:
        for arm, aggregations in ALL_ARMS:
            if arm not in active_arms:
                continue
            for aggregation in aggregations:
                if skip_lme and AGGREGATION_SPECS[aggregation]["needs_lme"]:
                    continue
                for band in bands:
                    jobs.append((npz_path, arm, aggregation, band))

    print(f"Total cells to evaluate: {len(jobs)}")

    all_rows: list[dict] = []
    results_path = out_dir / "results.csv"

    # Load existing rows when appending so checkpoints include them
    existing_df: pd.DataFrame | None = None
    if append and results_path.exists():
        existing_df = pd.read_csv(results_path)
        print(f"  Append mode: {len(existing_df)} existing rows in {results_path}")

    def _write_checkpoint(new_rows: list[dict]) -> None:
        df_new = pd.DataFrame(new_rows)
        if existing_df is not None:
            pd.concat([existing_df, df_new], ignore_index=True).to_csv(results_path, index=False)
        else:
            df_new.to_csv(results_path, index=False)

    start = time.time()
    for npz_path, arm, aggregation, band in tqdm(jobs, desc="Evaluating cells"):
        try:
            rows = evaluate_cell(
                npz_path=npz_path,
                manifest=manifest,
                arm=arm,
                aggregation=aggregation,
                band=band,
                cv_config=cv_config,
            )
            all_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(
                f"  SKIP {Path(npz_path).name} arm={arm} agg={aggregation} band={band}: {exc}"
            )

        # Checkpoint every 50 cells
        if len(all_rows) > 0 and (len(all_rows) % 50 == 0 or npz_path == jobs[-1][0]):
            _write_checkpoint(all_rows)

    # Final write
    if all_rows:
        _write_checkpoint(all_rows)
        elapsed = time.time() - start
        print(f"\nDone. {len(all_rows)} new rows written to {results_path}")
        print(f"Elapsed: {elapsed/60:.1f} min")
    else:
        print("No results produced.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run longitudinal aggregation factorial experiment.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parents[2],
        help="Path to project root (default: two levels above this script)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config YAML file (default: <project-root>/brainage_agg/config.yaml)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        metavar="ARM",
        help="Arms to evaluate (e.g. --arms B2). Default: all arms (A B B2).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to existing results.csv instead of overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on a small subject subset with only 3 extractor files for testing.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=30,
        help="Number of subjects to use in dry-run mode.",
    )
    parser.add_argument(
        "--skip-lme",
        action="store_true",
        help="Skip LME-based aggregations (lme_slope, lme_slope_change) for faster runs.",
    )
    args = parser.parse_args()
    run(
        project_root=args.project_root,
        config_path=args.config,
        dry_run=args.dry_run,
        n_dry_subjects=args.n_subjects,
        skip_lme=args.skip_lme,
        arms=args.arms,
        append=args.append,
    )


if __name__ == "__main__":
    main()
