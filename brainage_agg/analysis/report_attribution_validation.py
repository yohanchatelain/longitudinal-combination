"""Build the unified confirmatory voxel-attribution paper package."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attribution-validation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.validation.config import load_validation_config
from brainage_agg.validation.gates import evaluate_decision_gates
from brainage_agg.validation.metrics import (
    bootstrap_interval,
    boundary_bias_regression,
    map_similarity,
    stratified_roi_permutation,
)
from brainage_agg.validation.provenance import sha256_file
from brainage_agg.validation.statistics import maxT_pseudo_null_calibration


def _load_campaign(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "completion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing campaign manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Campaign is not complete: {run_dir} (status={manifest.get('status')})")
    planned = {json.dumps(cell, sort_keys=True) for cell in manifest["planned_cells"]}
    completed = {json.dumps(cell, sort_keys=True) for cell in manifest["completed_cells"]}
    if planned != completed:
        raise RuntimeError(f"Campaign cell mismatch: {run_dir}")
    return manifest


def _cell_manifests(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted(run_dir.glob("cells/**/cell_manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete cell manifest: {path}")
        for artifact in manifest["artifacts"]:
            artifact_path = path.parent / artifact["path"]
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"Missing or modified cell artifact: {artifact_path}")
        result.append((path.parent, manifest))
    return result


def _localization_summary(cell_manifests: list[tuple[Path, dict[str, Any]]], bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat([pd.read_csv(directory / "metrics.csv", dtype={"subject_id": str}) for directory, _ in cell_manifests], ignore_index=True)
    group_cols = ["cohort", "architecture", "target", "amplitude", "amplitude_role", "aggregation", "weight_type"]
    rows = []
    metrics = ["voxel_auroc", "normalized_auprc_lift", "attribution_mass_fraction", "bilateral_top5_hit"]
    for keys, frame in raw.groupby(group_cols, dropna=False):
        metadata = dict(zip(group_cols, keys))
        # Average repeated subjects across seeds first, then bootstrap subjects.
        subject = frame.groupby("subject_id")[metrics].mean()
        for metric in metrics:
            estimate, lower, upper = bootstrap_interval(
                subject[metric].to_numpy(), n_bootstrap=bootstrap, seed=seed
            )
            rows.append({**metadata, "analysis": "localization", "metric": metric, "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "n": len(subject)})
    summary = pd.DataFrame(rows)
    gate = summary.pivot_table(index=group_cols, columns="metric", values=["estimate", "ci_lower"]).reset_index()
    gate.columns = ["_".join(part for part in column if part).strip("_") if isinstance(column, tuple) else column for column in gate.columns]
    gate = gate.rename(
        columns={
            "estimate_voxel_auroc": "voxel_auroc",
            "ci_lower_voxel_auroc": "voxel_auroc_ci_lower",
            "estimate_normalized_auprc_lift": "normalized_auprc_lift",
            "ci_lower_normalized_auprc_lift": "normalized_auprc_lift_ci_lower",
            "estimate_bilateral_top5_hit": "bilateral_top5_hit_rate",
        }
    )
    return summary, gate


def _infer_null_metadata(path: Path) -> tuple[str, str]:
    lower = str(path).lower()
    cohort = "ppmi" if "ppmi" in lower else "nki"
    architecture = "cov_pool" if "cov_pool" in path.name else "double_conv"
    return cohort, architecture


def _null_summary(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, combined = [], {}
    for index, path in enumerate(paths):
        data = np.load(path, allow_pickle=False)
        permutation = np.asarray(data["permutation_abs"])
        if permutation.ndim == 3:
            permutation_for_calibration = permutation.mean(axis=0)
        elif permutation.ndim == 2:
            permutation_for_calibration = permutation
        else:
            raise ValueError(f"Unexpected permutation array shape in {path}: {permutation.shape}")
        calibration = maxT_pseudo_null_calibration(permutation_for_calibration)
        cohort, architecture = _infer_null_metadata(path)
        aggregation = next(
            (value for value in ("mean", "annualized_rate") if value in path.parts),
            "unknown",
        )
        weight_match = re.search(r"__(tstat|shap|hybrid)__", path.name)
        weight_type = weight_match.group(1) if weight_match else "unknown"
        rows.append({"analysis": "null_calibration", "cohort": cohort, "architecture": architecture, "aggregation": aggregation, "weight_type": weight_type, **calibration})
        prefix = f"cell_{index:04d}"
        for key in data.files:
            combined[f"{prefix}__{key}"] = data[key]
        combined[f"{prefix}__source"] = np.array(str(path))
    return pd.DataFrame(rows), combined


def _specificity_summary(
    cell_manifests: list[tuple[Path, dict[str, Any]]],
    run_dirs: list[Path],
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    for directory, manifest in cell_manifests:
        cell = manifest["cell"]
        if not np.isclose(float(cell["amplitude"]), float(manifest["primary_amplitude"])):
            continue
        for weight_type in cell["weight_types"]:
            records.append({**cell, "weight_type": weight_type, "path": directory / f"voxel_map_abs_{weight_type}.nii.gz"})
    frame = pd.DataFrame(records)
    rows = []
    rng = np.random.default_rng(seed)
    if frame.empty:
        return pd.DataFrame(columns=["cohort", "architecture", "target", "comparison", "estimate", "ci_lower", "ci_upper"])
    group_cols = ["cohort", "architecture", "aggregation", "weight_type"]
    for keys, group in frame.groupby(group_cols):
        cohort, architecture, aggregation, weight_type = keys
        targets = sorted(group["target"].unique())
        loaded = {(row.target, int(row.seed)): np.asarray(nib.load(str(row.path)).dataobj, dtype=np.float32) for row in group.itertuples()}
        prior = {}
        for run_dir in run_dirs:
            for path in run_dir.glob(f"controls/{cohort}/{architecture}/seed_*/voxel_map_abs_unit_mean_brain.nii.gz"):
                match = re.search(r"seed_(\d+)", str(path))
                if match:
                    prior[int(match.group(1))] = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        for target in targets:
            seeds = sorted(int(x) for x in group[group["target"] == target]["seed"].unique())
            within, between, prior_similarity = [], [], []
            other_maps = [value for (other_target, _), value in loaded.items() if other_target != target]
            for current_seed in seeds:
                current = loaded[(target, current_seed)]
                same = [loaded[(target, other_seed)] for other_seed in seeds if other_seed != current_seed]
                within.append(float(np.nanmean([map_similarity(current, value) for value in same])))
                between.append(float(np.nanmean([map_similarity(current, value) for value in other_maps])))
                prior_similarity.append(map_similarity(current, prior[current_seed]) if current_seed in prior else np.nan)
            for comparison, difference in (
                ("within_minus_between", np.asarray(within) - np.asarray(between)),
                ("within_minus_prior", np.asarray(within) - np.asarray(prior_similarity)),
            ):
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    draws = [np.mean(rng.choice(finite, len(finite), replace=True)) for _ in range(bootstrap)]
                    estimate, lower, upper = float(np.mean(finite)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
                else:
                    estimate = lower = upper = float("nan")
                rows.append(
                    {
                        "analysis": "specificity",
                        "cohort": cohort,
                        "architecture": architecture,
                        "aggregation": aggregation,
                        "weight_type": weight_type,
                        "target": target,
                        "comparison": comparison,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n": len(finite),
                    }
                )
    return pd.DataFrame(rows)


def _roi_summary(paths: list[Path], bootstrap: int, seed: int) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["analysis", "cohort", "architecture", "score_type", "estimate", "ci_lower", "ci_upper", "n"])
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"cohort", "architecture", "seed", "score_type", "voxel_score", "roi_effect"}
    if missing := required - set(raw):
        raise ValueError(f"ROI concordance input missing columns: {sorted(missing)}")
    rows = []
    for keys, frame in raw.groupby(["cohort", "architecture", "score_type"]):
        seed_correlations = []
        for _, seed_frame in frame.groupby("seed"):
            seed_correlations.append(seed_frame["voxel_score"].corr(seed_frame["roi_effect"]))
        estimate, lower, upper = bootstrap_interval(seed_correlations, n_bootstrap=bootstrap, seed=seed)
        row = {"analysis": "roi_concordance", "cohort": keys[0], "architecture": keys[1], "score_type": keys[2], "metric": "pearson_r", "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "n": len(seed_correlations)}
        strata_columns = ["hemisphere", "tissue_class", "region_volume_quintile"]
        if set(strata_columns) <= set(frame):
            # Permutation inference is region-level and stratified jointly by
            # all predeclared nuisance categories. Average scores across seeds
            # first; seeds remain the bootstrap unit for the interval above.
            region_columns = [column for column in ("label_id", "region_name") if column in frame]
            averaged = frame.groupby(region_columns + strata_columns, dropna=False)[["voxel_score", "roi_effect"]].mean().reset_index()
            perm = stratified_roi_permutation(
                averaged["voxel_score"].to_numpy(),
                averaged["roi_effect"].to_numpy(),
                averaged[strata_columns],
                seed=seed,
            )
            row["stratified_empirical_p_two_sided"] = perm["empirical_p_two_sided"]
        rows.append(row)
    return pd.DataFrame(rows)


def _empirical_map_specificity(paths: list[Path], bootstrap: int, seed: int) -> pd.DataFrame:
    """Reanalyze existing age/sex/severity/unit-prior seed maps."""
    if not paths:
        return pd.DataFrame()
    frames = []
    for csv_path in paths:
        frame = pd.read_csv(csv_path)
        if "map_path" not in frame or "prior_path" not in frame:
            raise ValueError("Empirical map manifest requires map_path and prior_path columns")
        frame["map_path"] = frame["map_path"].map(lambda value: str((csv_path.parent / value).resolve()) if not Path(value).is_absolute() else value)
        frame["prior_path"] = frame["prior_path"].map(lambda value: str((csv_path.parent / value).resolve()) if not Path(value).is_absolute() else value)
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    required = {"cohort", "architecture", "target", "seed", "map_path", "prior_path"}
    if missing := required - set(manifest):
        raise ValueError(f"Empirical map manifest missing columns: {sorted(missing)}")
    rows = []
    rng = np.random.default_rng(seed)
    for (cohort, architecture), group in manifest.groupby(["cohort", "architecture"]):
        targets = sorted(group["target"].unique())
        loaded = {(row.target, int(row.seed)): np.asarray(nib.load(row.map_path).dataobj, dtype=np.float32) for row in group.itertuples()}
        prior = {int(row.seed): np.asarray(nib.load(row.prior_path).dataobj, dtype=np.float32) for row in group.itertuples()}
        for target in targets:
            seeds = sorted(int(value) for value in group[group["target"] == target]["seed"].unique())
            within, between, prior_values = [], [], []
            other = [value for (name, _), value in loaded.items() if name != target]
            for current_seed in seeds:
                current = loaded[(target, current_seed)]
                same = [loaded[(target, other_seed)] for other_seed in seeds if other_seed != current_seed]
                within.append(np.nanmean([map_similarity(current, value) for value in same]))
                between.append(np.nanmean([map_similarity(current, value) for value in other]))
                prior_values.append(map_similarity(current, prior[current_seed]))
            for comparison, differences in (
                ("within_minus_between", np.asarray(within) - np.asarray(between)),
                ("within_minus_prior", np.asarray(within) - np.asarray(prior_values)),
            ):
                finite = differences[np.isfinite(differences)]
                draws = [np.mean(rng.choice(finite, len(finite), replace=True)) for _ in range(bootstrap)] if len(finite) else [np.nan]
                rows.append(
                    {
                        "analysis": "empirical_target_specificity",
                        "cohort": cohort,
                        "architecture": architecture,
                        "target": target,
                        "comparison": comparison,
                        "estimate": float(np.mean(finite)) if len(finite) else np.nan,
                        "ci_lower": float(np.nanquantile(draws, 0.025)),
                        "ci_upper": float(np.nanquantile(draws, 0.975)),
                        "n": len(finite),
                    }
                )
    return pd.DataFrame(rows)


def _boundary_summary(cell_manifests: list[tuple[Path, dict[str, Any]]], run_dirs: list[Path]) -> pd.DataFrame:
    boundary_by_scope = {}
    for run_dir in run_dirs:
        for path in run_dir.glob("controls/*/*/seed_*/region_boundary_metrics.csv"):
            parts = path.relative_to(run_dir).parts
            boundary_by_scope.setdefault((parts[1], parts[2]), pd.read_csv(path))
    tables = []
    for directory, manifest in cell_manifests:
        cell = manifest["cell"]
        if not np.isclose(float(cell["amplitude"]), float(manifest["primary_amplitude"])):
            continue
        regional_path = directory / "regional_stats.csv"
        if not regional_path.is_file():
            continue
        frame = pd.read_csv(regional_path)
        frame["cohort"] = cell["cohort"]
        frame["architecture"] = cell["architecture"]
        frame["target"] = cell["target"]
        frame["aggregation"] = cell["aggregation"]
        tables.append(frame)
    if not tables:
        return pd.DataFrame()
    regional = pd.concat(tables, ignore_index=True)
    rows = []
    for keys, frame in regional.groupby(["cohort", "architecture", "target", "aggregation", "weight_type"]):
        boundary = boundary_by_scope.get((keys[0], keys[1]))
        if boundary is None:
            continue
        mean_region = frame.groupby("label_id", as_index=False)["abs_importance"].mean()
        try:
            coefficients = boundary_bias_regression(mean_region, boundary)
        except ValueError:
            continue
        for row in coefficients.itertuples():
            rows.append(
                {
                    "analysis": "boundary_bias",
                    "cohort": keys[0], "architecture": keys[1], "target": keys[2],
                    "aggregation": keys[3], "weight_type": keys[4], "metric": row.term,
                    "estimate": row.estimate, "model_r_squared": row.r_squared,
                    "n": row.n_regions,
                }
            )
    return pd.DataFrame(rows)


def _control_summary(run_dirs: list[Path], bootstrap: int, seed: int) -> pd.DataFrame:
    records = []
    for run_dir in run_dirs:
        for mean_path in run_dir.glob("controls/*/*/seed_*/voxel_map_abs_unit_mean_brain.nii.gz"):
            relative = mean_path.relative_to(run_dir).parts
            cohort, architecture = relative[1], relative[2]
            seed_match = re.search(r"seed_(\d+)", relative[3])
            if not seed_match:
                continue
            constant_path = mean_path.with_name("voxel_map_abs_unit_constant_masked.nii.gz")
            phase_path = mean_path.with_name("voxel_map_abs_unit_phase_scrambled.nii.gz")
            if not constant_path.is_file() or not phase_path.is_file():
                continue
            arrays = {
                "mean_brain": np.asarray(nib.load(str(mean_path)).dataobj, dtype=np.float32),
                "constant_masked": np.asarray(nib.load(str(constant_path)).dataobj, dtype=np.float32),
                "phase_scrambled": np.asarray(nib.load(str(phase_path)).dataobj, dtype=np.float32),
            }
            for first, second in (("mean_brain", "constant_masked"), ("mean_brain", "phase_scrambled"), ("constant_masked", "phase_scrambled")):
                records.append(
                    {
                        "cohort": cohort,
                        "architecture": architecture,
                        "seed": int(seed_match.group(1)),
                        "comparison": f"{first}_vs_{second}",
                        "similarity": map_similarity(arrays[first], arrays[second]),
                    }
                )
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame(records)
    rows = []
    for keys, frame in raw.groupby(["cohort", "architecture", "comparison"]):
        estimate, lower, upper = bootstrap_interval(
            frame["similarity"].to_numpy(), n_bootstrap=bootstrap, seed=seed
        )
        rows.append(
            {
                "analysis": "input_prior_control",
                "cohort": keys[0], "architecture": keys[1], "comparison": keys[2],
                "metric": "map_similarity", "estimate": estimate,
                "ci_lower": lower, "ci_upper": upper, "n": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _figures(output: Path, localization: pd.DataFrame, null: pd.DataFrame, specificity: pd.DataFrame, roi: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = localization[(localization["metric"] == "voxel_auroc") & (localization["amplitude_role"] == "primary")]
    if data.empty:
        _placeholder(ax, "No completed localization estimates")
    else:
        labels = data["cohort"] + "/" + data["architecture"] + "/" + data["target"]
        ax.errorbar(np.arange(len(data)), data["estimate"], yerr=[data["estimate"] - data["ci_lower"], data["ci_upper"] - data["estimate"]], fmt="o")
        ax.axhline(0.75, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(data)), labels, rotation=80, ha="right", fontsize=7)
        ax.set_ylabel("Voxel AUROC")
    fig.tight_layout(); fig.savefig(output / "figure_injected_localization.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if null.empty:
        _placeholder(ax, "No completed permutation-null estimates")
    else:
        labels = null["cohort"] + "/" + null["architecture"]
        ax.errorbar(np.arange(len(null)), null["maxT_fwer"], yerr=[null["maxT_fwer"] - null["maxT_fwer_ci_lower"], null["maxT_fwer_ci_upper"] - null["maxT_fwer"]], fmt="o")
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(null)), labels, rotation=45, ha="right")
        ax.set_ylabel("LOPO maxT family-wise false-positive rate")
    fig.tight_layout(); fig.savefig(output / "figure_null_calibration.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if specificity.empty:
        _placeholder(ax, "No completed target/prior specificity estimates")
    else:
        labels = specificity["target"] + "/" + specificity["comparison"].str.replace("within_minus_", "")
        ax.errorbar(np.arange(len(specificity)), specificity["estimate"], yerr=[specificity["estimate"] - specificity["ci_lower"], specificity["ci_upper"] - specificity["estimate"]], fmt="o")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(np.arange(len(specificity)), labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Paired similarity difference")
    fig.tight_layout(); fig.savefig(output / "figure_target_specificity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if roi.empty:
        _placeholder(ax, "ROI concordance unavailable (secondary analysis)")
    else:
        labels = roi["cohort"] + "/" + roi["score_type"]
        ax.errorbar(np.arange(len(roi)), roi["estimate"], yerr=[roi["estimate"] - roi["ci_lower"], roi["ci_upper"] - roi["estimate"]], fmt="o")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(np.arange(len(roi)), labels, rotation=45, ha="right")
        ax.set_ylabel("Seed-bootstrap Pearson r")
    fig.tight_layout(); fig.savefig(output / "figure_roi_concordance.png", dpi=180); plt.close(fig)


def _manuscript(decision: dict[str, Any], roi: pd.DataFrame) -> str:
    roi_sentence = "ROI concordance was analyzed as a secondary outcome and was not used in any decision gate."
    if not roi.empty:
        roi_sentence += " Raw, null-excess, and empirical-standardized estimates are reported with seed-bootstrap intervals."
    return f"""# Confirmatory voxel-attribution validation

## Methods

The prior 36-cell, six-aggregation, ten-seed sweep was designated exploratory. Confirmatory evaluation used disjoint pilot and validation subjects in NKI and PPMI. Smooth multiplicative attenuation was injected into bilateral putamen, hippocampus, superior-frontal cortex, or cerebellar white matter in raw T1 volumes before scaling; all T1, local-rank, and Sobel channels were then recomputed. The primary attenuation was selected on the pilot split as the candidate nearest regional Cohen's d=0.8 without inspecting attribution maps. Localization was evaluated with voxel AUROC, prevalence-normalized AUPRC, attribution mass, bilateral target rank, and top-five hits.

The cohort null retained the regional statistic from every label permutation. One-sided empirical p-values used the Monte Carlo +1 correction, and family-wise p-values used the permutation maximum across regions. Leave-one-permutation-out pseudo-null evaluations assessed maxT calibration. Unit-weight mean-brain, constant masked, and phase-scrambled inputs quantified the architecture/input prior. Boundary bias was modeled using boundary-voxel fraction, log region volume, and tissue class rather than named-region exclusions. {roi_sentence}

## Results

The machine-evaluated claim is: **{decision['permitted_claim']}** The previously missing NKI `lme_slope/hybrid` exploratory cell is now treated as completed output; it does not alter the confirmatory gates and is not used as positive biological validation.

## Limitations

FreeSurfer ROI effects and voxel attribution interrogate different models, so disagreement cannot by itself identify which representation is biologically correct. Conversely, ROI agreement cannot validate voxel localization. Biological interpretation is withheld unless positive-control localization, maxT calibration, and target specificity pass in the relevant cohort and architecture. The cerebellar injection is an adversarial edge-prior control, not evidence for cerebellar biology. General claims additionally require reproducibility in both cohorts and both architectures.
"""


def build_report(
    run_dirs: list[Path],
    output: Path,
    *,
    config_path: Path,
    null_paths: list[Path],
    roi_paths: list[Path],
    empirical_map_paths: list[Path] | None = None,
) -> dict[str, Any]:
    config = load_validation_config(config_path)
    for run_dir in run_dirs:
        _load_campaign(run_dir)
    cells = [cell for run_dir in run_dirs for cell in _cell_manifests(run_dir)]
    if not cells:
        raise RuntimeError("No complete injection cells were found")
    bootstrap = int(config["controls"]["bootstrap_samples"])
    bootstrap_seed = int(config["controls"]["bootstrap_seed"])
    localization, localization_gate = _localization_summary(cells, bootstrap, bootstrap_seed)
    discovered_null = list(null_paths)
    for run_dir in run_dirs:
        discovered_null.extend(run_dir.glob("**/permutation_region_stats_*.npz"))
    null, combined_null = _null_summary(sorted(set(discovered_null)))
    specificity = _specificity_summary(cells, run_dirs, bootstrap=bootstrap, seed=bootstrap_seed)
    empirical_specificity = _empirical_map_specificity(
        empirical_map_paths or [], bootstrap, bootstrap_seed
    )
    boundary = _boundary_summary(cells, run_dirs)
    control = _control_summary(run_dirs, bootstrap, bootstrap_seed)
    roi = _roi_summary(roi_paths, bootstrap, bootstrap_seed)

    null_gate = null[["cohort", "architecture", "maxT_fwer", "maxT_fwer_ci_upper"]].copy() if not null.empty else pd.DataFrame(columns=["cohort", "architecture", "maxT_fwer", "maxT_fwer_ci_upper"])
    decision = evaluate_decision_gates(
        localization_gate,
        null_gate,
        specificity,
        config["decision_gates"],
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(
        [localization, null, specificity, empirical_specificity, control, boundary, roi],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(output / "validation_summary.csv", index=False)
    np.savez_compressed(output / "permutation_region_stats.npz", **combined_null)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    _figures(output, localization, null, specificity, roi)
    (output / "manuscript_attribution_validation.md").write_text(_manuscript(decision, roi))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "attribution_validation.yaml")
    parser.add_argument("--null-npz", type=Path, action="append", default=[])
    parser.add_argument("--roi-concordance-csv", type=Path, action="append", default=[])
    parser.add_argument(
        "--empirical-map-csv", type=Path, action="append", default=[],
        help="Manifest(s) for age/sex/severity/unit-prior seed-map reanalysis.",
    )
    args = parser.parse_args()
    decision = build_report(
        args.run_dir,
        args.output,
        config_path=args.config,
        null_paths=args.null_npz,
        roi_paths=args.roi_concordance_csv,
        empirical_map_paths=args.empirical_map_csv,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
