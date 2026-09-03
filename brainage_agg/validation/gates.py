"""Machine-evaluated preregistered decision gates and claim policy."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def _all_true(values: pd.Series) -> bool:
    return bool(len(values) > 0 and values.fillna(False).astype(bool).all())


def evaluate_decision_gates(
    localization: pd.DataFrame,
    null_calibration: pd.DataFrame,
    specificity: pd.DataFrame,
    thresholds: Mapping[str, Mapping[str, float]],
    *,
    expected_cohorts: tuple[str, ...] = ("nki", "ppmi"),
    expected_architectures: tuple[str, ...] = ("double_conv", "cov_pool"),
) -> dict[str, object]:
    """Evaluate cell-level gates and derive the strongest permitted claim."""
    required_loc = {
        "cohort", "architecture", "target", "amplitude_role", "voxel_auroc",
        "voxel_auroc_ci_lower", "normalized_auprc_lift",
        "normalized_auprc_lift_ci_lower", "bilateral_top5_hit_rate",
    }
    missing = required_loc - set(localization.columns)
    if missing:
        raise ValueError(f"Localization table missing columns: {sorted(missing)}")
    primary = localization[localization["amplitude_role"] == "primary"].copy()
    loc_cfg = thresholds["localization"]
    primary["gate_pass"] = (
        (primary["voxel_auroc"] >= loc_cfg["voxel_auroc_min"])
        & (primary["voxel_auroc_ci_lower"] > loc_cfg["voxel_auroc_lower_min"])
        & (primary["normalized_auprc_lift"] >= loc_cfg["normalized_auprc_lift_min"])
        & (primary["normalized_auprc_lift_ci_lower"] > loc_cfg["normalized_auprc_lift_lower_min"])
        & (primary["bilateral_top5_hit_rate"] >= loc_cfg["bilateral_top5_hit_rate_min"])
    )

    null_cfg = thresholds["null_calibration"]
    null = null_calibration.copy()
    null["gate_pass"] = (
        (null["maxT_fwer"] <= null_cfg["maxT_fwer_max"])
        & (null["maxT_fwer_ci_upper"] <= null_cfg["maxT_fwer_upper_max"])
    )

    spec_cfg = thresholds["specificity"]
    specific = specificity.copy()
    specific["gate_pass"] = specific["ci_lower"] > spec_cfg["paired_difference_lower_min"]

    cell_results = []
    for cohort in expected_cohorts:
        for architecture in expected_architectures:
            loc_cell = primary[(primary["cohort"] == cohort) & (primary["architecture"] == architecture)]
            null_cell = null[(null["cohort"] == cohort) & (null["architecture"] == architecture)]
            # The preregistered cohort-null architecture is double_conv; its
            # calibration gate is shared with the injected-effect cov_pool
            # control rather than silently demanding an unregistered null run.
            if null_cell.empty:
                null_cell = null[
                    (null["cohort"] == cohort)
                    & (null["architecture"] == "double_conv")
                ]
            spec_cell = specific[(specific["cohort"] == cohort) & (specific["architecture"] == architecture)]
            complete = not loc_cell.empty and not null_cell.empty and not spec_cell.empty
            passed = complete and _all_true(loc_cell["gate_pass"]) and _all_true(null_cell["gate_pass"]) and _all_true(spec_cell["gate_pass"])
            cell_results.append(
                {
                    "cohort": cohort,
                    "architecture": architecture,
                    "complete": bool(complete),
                    "localization_pass": _all_true(loc_cell["gate_pass"]),
                    "null_calibration_pass": _all_true(null_cell["gate_pass"]),
                    "specificity_pass": _all_true(spec_cell["gate_pass"]),
                    "all_gates_pass": bool(passed),
                }
            )

    all_pass = all(row["all_gates_pass"] for row in cell_results)
    passing = [f"{row['cohort']}/{row['architecture']}" for row in cell_results if row["all_gates_pass"]]
    localization_or_calibration_failed = any(
        not row["localization_pass"] or not row["null_calibration_pass"] for row in cell_results
    )
    if all_pass:
        permitted_claim = (
            "Technically validated, target-specific voxel localization in both cohorts and architectures; "
            "biological concordance remains a separate empirical result."
        )
        claim_level = "general"
    elif passing:
        permitted_claim = "Technically validated only for: " + ", ".join(passing) + "."
        claim_level = "scoped"
    elif localization_or_calibration_failed:
        permitted_claim = (
            "Voxel attribution may be presented only as an architecture/preprocessing diagnostic; "
            "highlighted regions must not be interpreted biologically."
        )
        claim_level = "diagnostic_only"
    else:
        permitted_claim = "No localization claim is permitted because confirmatory evidence is incomplete."
        claim_level = "none"
    return {
        "schema_version": 1,
        "all_gates_pass": bool(all_pass),
        "claim_level": claim_level,
        "permitted_claim": permitted_claim,
        "passing_scopes": passing,
        "cells": cell_results,
        "roi_concordance_used_as_gate": False,
    }
