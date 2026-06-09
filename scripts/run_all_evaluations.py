#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    evaluate_classification,
    evaluate_regression,
    feature_metadata,
    make_crosssectional_matrix,
    make_longitudinal_matrix,
)
from src.make_longitudinal_pairs import make_all_forward_pairs
from src.utils import ensure_dir, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--feature-glob", default="*.npz")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _adult_vs_child_label(studies_col: pd.Series) -> pd.Series:
    """Return 'adult'/'child'/NaN from a set-valued studies column."""
    has_adult = studies_col.str.contains("Longitudinal_Adult", na=False)
    has_child = studies_col.str.contains("Longitudinal_Child", na=False)
    label = pd.Series(index=studies_col.index, dtype=object)
    label[has_adult] = "adult"
    label[has_child] = "child"
    return label


def _process_file(feature_file, seeds, test_size, rf_params, long_modes):
    feature_file = Path(feature_file)
    print(f"Evaluating {feature_file.name}", flush=True)
    X, meta_df = make_crosssectional_matrix(feature_file)
    feature_meta = feature_metadata(feature_file)
    groups = meta_df["id"].values

    rows = []
    rows.extend(
        evaluate_regression(
            X,
            meta_df["age"].values,
            groups,
            seeds,
            feature_file.name,
            feature_meta,
            task="age",
            analysis_type="crosssectional",
            pair_mode="",
            test_size=test_size,
            rf_params=rf_params,
        )
    )
    rows.extend(
        evaluate_classification(
            X,
            meta_df["sex"].values,
            groups,
            seeds,
            feature_file.name,
            feature_meta,
            task="sex",
            analysis_type="crosssectional",
            pair_mode="",
            test_size=test_size,
            rf_params=rf_params,
        )
    )
    for optional in ["session_type", "study_num"]:
        if optional in meta_df.columns:
            rows.extend(
                evaluate_classification(
                    X,
                    meta_df[optional].values,
                    groups,
                    seeds,
                    feature_file.name,
                    feature_meta,
                    task=optional,
                    analysis_type="crosssectional",
                    pair_mode="",
                    test_size=test_size,
                    rf_params=rf_params,
                )
            )

    if "studies" in meta_df.columns:
        adult_vs_child = _adult_vs_child_label(meta_df["studies"])
        valid = adult_vs_child.notna().to_numpy()
        rows.extend(
            evaluate_classification(
                X[valid],
                adult_vs_child[valid].values,
                groups[valid],
                seeds,
                feature_file.name,
                feature_meta,
                task="adult_vs_child",
                analysis_type="crosssectional",
                pair_mode="",
                test_size=test_size,
                rf_params=rf_params,
            )
        )

    pair_df = make_all_forward_pairs(meta_df)
    if not pair_df.empty:
        for mode in long_modes:
            X_pair, pair_meta = make_longitudinal_matrix(feature_file, pair_df, mode)
            pair_groups = pair_meta["id"].values
            rows.extend(
                evaluate_regression(
                    X_pair,
                    pair_meta["delta_age"].values,
                    pair_groups,
                    seeds,
                    feature_file.name,
                    feature_meta,
                    task="delta_age",
                    analysis_type="longitudinal",
                    pair_mode=mode,
                    test_size=test_size,
                    rf_params=rf_params,
                )
            )
            rows.extend(
                evaluate_classification(
                    X_pair,
                    pair_meta["sex"].values,
                    pair_groups,
                    seeds,
                    feature_file.name,
                    feature_meta,
                    task="sex",
                    analysis_type="longitudinal",
                    pair_mode=mode,
                    test_size=test_size,
                    rf_params=rf_params,
                )
            )

    return rows


def main():
    args = parse_args()
    config = load_config(args.config)
    feature_dir = Path(config["output"]["feature_dir"])
    result_dir = ensure_dir(config["output"]["result_dir"])
    out_csv = result_dir / "evaluation_results.csv"

    eval_cfg = config["evaluation"]
    rf_params = {
        "n_estimators": eval_cfg.get("n_estimators", 200),
        "max_depth": eval_cfg.get("max_depth", 6),
        "min_samples_split": eval_cfg.get("min_samples_split", 5),
    }
    seeds = config["seeds"]
    test_size = eval_cfg.get("test_size", 0.1)
    long_modes = ["delta", "concat_delta", "concat", "annualized_delta"]

    feature_files = sorted(feature_dir.glob(args.feature_glob))
    if not feature_files:
        raise RuntimeError(f"No feature files found in {feature_dir}")

    # Read the existing header (if any) so we can align new rows to the same schema.
    existing_cols: list[str] | None = None
    if out_csv.exists():
        existing_cols = pd.read_csv(out_csv, nrows=0).columns.tolist()

    write_header = existing_cols is None
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(
                _process_file,
                str(f),
                seeds,
                test_size,
                rf_params,
                long_modes,
            ): f
            for f in feature_files
        }
        for future in as_completed(futures):
            f = futures[future]
            try:
                rows = future.result()
                if rows:
                    chunk = pd.DataFrame(rows)
                    if existing_cols is not None:
                        # Add any columns the file doesn't know about yet, drop extras.
                        for col in existing_cols:
                            if col not in chunk.columns:
                                chunk[col] = ""
                        chunk = chunk[existing_cols]
                    else:
                        existing_cols = chunk.columns.tolist()
                    chunk.to_csv(out_csv, mode="a", header=write_header, index=False)
                    write_header = False
                    print(f"  -> wrote {len(rows)} rows for {f.name}", flush=True)
            except Exception as exc:
                print(f"ERROR processing {f.name}: {exc}", flush=True)

    print(f"Done. Results in {out_csv}")


if __name__ == "__main__":
    main()
