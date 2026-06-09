#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HIGHER_IS_BETTER = {"balanced_accuracy", "auc", "r2", "pearson", "spearman", "f1_macro"}
LOWER_IS_BETTER = {"mae"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results/evaluation_results.csv")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results = pd.read_csv(args.results)
    group_cols = [
        "scaler",
        "channels",
        "task",
        "analysis_type",
        "pair_mode",
        "model",
        "metric",
    ]
    summary = (
        results.groupby(group_cols, dropna=False)["value"]
        .agg(mean="mean", std="std", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count")
        .reset_index()
    )

    summary["rank"] = pd.NA
    rank_group_cols = ["task", "analysis_type", "pair_mode", "model", "metric"]
    for _, idx in summary.groupby(rank_group_cols, dropna=False).groups.items():
        metric = summary.loc[list(idx), "metric"].iloc[0]
        ascending = metric in LOWER_IS_BETTER
        if metric not in HIGHER_IS_BETTER and metric not in LOWER_IS_BETTER:
            continue
        summary.loc[list(idx), "rank"] = summary.loc[list(idx), "mean"].rank(
            method="min", ascending=ascending
        ).astype("Int64")

    out = Path(args.output) if args.output else Path(args.results).with_name("summary_by_feature_set.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
