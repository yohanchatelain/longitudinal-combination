"""Run preregistered positive-control cells for voxel-attribution validation.

The command intentionally runs one bounded cell at a time so a scheduler can
enforce at most two concurrent GPU tasks. Every cell is atomic: it either
writes a ``cell_manifest.json`` with all three weight methods, or exits nonzero.

Examples
--------
Prepare deterministic, disjoint assignments::

    python -m brainage_agg.experiment.run_attribution_validation prepare --run-id confirmatory_v1

Run one injected-effect cell::

    python -m brainage_agg.experiment.run_attribution_validation injection-cell \
      --run-id confirmatory_v1 --cohort nki --architecture double_conv \
      --target putamen --amplitude 0.05 --aggregation mean --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.voxel_attribution import load_freesurfer_lut, project_to_atlas
from brainage_agg.experiment.run_voxel_importance import (
    _attribute_one_subject,
    _build_cnn,
    _compute_seed_weights,
)
from brainage_agg.analysis.importance_weights import rank_normalize
from brainage_agg.validation.config import load_validation_config
from brainage_agg.validation.controls import constant_masked_input, phase_scramble_input, unit_weight_prior
from brainage_agg.validation.injection import (
    apply_multiplicative_attenuation,
    deterministic_subject_assignment,
    select_calibrated_amplitude,
    tapered_label_mask,
)
from brainage_agg.validation.metrics import localization_metrics, region_boundary_metrics
from brainage_agg.validation.provenance import environment_snapshot, git_state, sha256_file
from src.io import load_cohort, load_mgz_float32, load_mgz_mask
from src.preprocessing import build_channels

WEIGHT_TYPES = ["tstat", "shap", "hybrid"]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def _dataset(config: dict[str, Any], cohort_name: str) -> tuple[pd.DataFrame, Path, str]:
    cohort_entry = config["data"]["cohorts"][cohort_name]
    brain_config_path = _resolve(cohort_entry["config"])
    with brain_config_path.open() as handle:
        brain_config = yaml.safe_load(handle)
    cohort_csv = _resolve(brain_config["data"]["cohort_csv"])
    fs_root = _resolve(brain_config["data"]["fs_root"])
    cohort = load_cohort(str(cohort_csv), freesurfer_root=str(fs_root))
    return cohort, fs_root, config["data"]["parcellation"]


def _parcellation_path(fs_root: Path, directory: str, filename: str) -> Path:
    return fs_root / str(directory) / "mri" / filename


def _eligible_subjects(cohort: pd.DataFrame, fs_root: Path, parc_filename: str) -> list[str]:
    valid = cohort[cohort.apply(lambda row: _parcellation_path(fs_root, row["directory"], parc_filename).is_file(), axis=1)]
    # Both confirmatory aggregations are run on one frozen population, so all
    # selected subjects must have at least two visits and positive time span.
    eligible = []
    for subject_id, rows in valid.groupby("id"):
        ages = np.sort(rows["age"].astype(float).unique())
        if len(ages) >= 2 and ages[-1] > ages[0]:
            eligible.append(str(subject_id))
    return eligible


def _planned_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    injection = config["injection"]
    cells = []
    for cohort in sorted(config["data"]["cohorts"]):
        for architecture in injection["architectures"]:
            for target in sorted(injection["targets"]):
                for amplitude in injection["amplitudes"]:
                    for aggregation in injection["aggregations"]:
                        for seed in injection["seeds"]:
                            cells.append(
                                {
                                    "kind": "injection",
                                    "cohort": cohort,
                                    "architecture": architecture,
                                    "target": target,
                                    "amplitude": float(amplitude),
                                    "aggregation": aggregation,
                                    "seed": int(seed),
                                    "weight_types": list(injection["weight_types"]),
                                }
                            )
        for architecture in injection["architectures"]:
            for seed in injection["seeds"]:
                cells.append(
                    {
                        "kind": "control",
                        "cohort": cohort,
                        "architecture": architecture,
                        "seed": int(seed),
                        "inputs": list(config["controls"]["unit_weight_inputs"]),
                    }
                )
        for aggregation in config["cohort_null"]["aggregations"]:
            cells.append(
                {
                    "kind": "null",
                    "cohort": cohort,
                    "architecture": config["cohort_null"]["architecture"],
                    "aggregation": aggregation,
                    "seeds": list(config["cohort_null"]["seeds"]),
                    "weight_types": list(config["cohort_null"]["weight_types"]),
                    "n_subjects": int(config["cohort_null"]["n_subjects"]),
                    "n_permutations": int(config["cohort_null"]["n_permutations"]),
                }
            )
    return cells


def prepare(config_path: Path, run_id: str) -> Path:
    config = load_validation_config(config_path)
    run_root = _resolve(config["output"]["root"]) / run_id
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_root}")
    run_root.mkdir(parents=True)
    assignments = []
    all_subject_ids = []
    injection = config["injection"]
    for cohort_name in sorted(config["data"]["cohorts"]):
        cohort, fs_root, parc = _dataset(config, cohort_name)
        eligible = _eligible_subjects(cohort, fs_root, parc)
        assigned = deterministic_subject_assignment(
            eligible,
            n_validation=int(injection["validation_subjects"]),
            n_pilot=int(injection["pilot_subjects"]),
            seed=int(injection["assignment_seed"]),
        )
        assigned.insert(0, "cohort", cohort_name)
        assignments.append(assigned)
        all_subject_ids.extend(f"{cohort_name}:{sid}" for sid in assigned["subject_id"])
    assignment_df = pd.concat(assignments, ignore_index=True)
    assignment_df.to_csv(run_root / "subject_assignment.csv", index=False)
    (run_root / "subject_ids.json").write_text(json.dumps(sorted(all_subject_ids), indent=2) + "\n")
    with (run_root / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (run_root / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(ROOT),
        "config_sha256": sha256_file(run_root / "resolved_config.yaml"),
        "assignment_sha256": sha256_file(run_root / "subject_assignment.csv"),
        "rng_seeds": {
            "subject_assignment": injection["assignment_seed"],
            "cnn": injection["seeds"],
            "null": config["cohort_null"]["permutation_seed"],
            "bootstrap": config["controls"]["bootstrap_seed"],
        },
        "planned_cells": _planned_cells(config),
        "completed_cells": [],
        "failed_cells": [],
    }
    _atomic_json(run_root / "completion_manifest.json", manifest)
    return run_root


def _load_raw(row: pd.Series, fs_root: Path, parc_filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_path = Path(str(row["image_path"]))
    image_obj = nib.load(str(image_path))
    image = np.asarray(image_obj.dataobj, dtype=np.float32)
    mask_path = Path(str(row.get("mask_path", "")))
    mask = load_mgz_mask(str(mask_path)) if mask_path.is_file() else image > 0
    atlas_path = _parcellation_path(fs_root, str(row["directory"]), parc_filename)
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Missing parcellation: {atlas_path}")
    atlas = np.asarray(nib.load(str(atlas_path)).dataobj, dtype=np.int32)
    if image.shape != mask.shape or image.shape != atlas.shape:
        raise ValueError(f"Image/mask/atlas shapes differ for {row['directory']}")
    return image, mask, atlas, image_obj.affine


def _target_labels(target: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    left = [int(label) for label in target["left_labels"]]
    right = [int(label) for label in target["right_labels"]]
    return left, right, left + right


def _pilot_regional_values(
    cohort: pd.DataFrame,
    assigned: pd.DataFrame,
    fs_root: Path,
    parc_filename: str,
    labels: list[int],
    amplitudes: list[float],
    boundary_voxels: int,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    pilot = assigned[assigned["split"] == "pilot"]
    group_by_id = pilot.set_index("subject_id")["synthetic_group"].to_dict()
    control, injected = [], {float(a): [] for a in amplitudes}
    selected = cohort[cohort["_subject_id"].isin(group_by_id)]
    for subject_id, rows in selected.groupby("_subject_id"):
        values = {float(a): [] for a in amplitudes}
        untreated = []
        for _, row in rows.iterrows():
            image, _, atlas, _ = _load_raw(row, fs_root, parc_filename)
            field = tapered_label_mask(atlas, labels, boundary_voxels)
            region = field > 0
            untreated.append(float(image[region].mean()))
            for amplitude in amplitudes:
                modified = apply_multiplicative_attenuation(image, field, float(amplitude))
                values[float(amplitude)].append(float(modified[region].mean()))
        if group_by_id[str(subject_id)] == "control":
            control.append(float(np.mean(untreated)))
        else:
            for amplitude in amplitudes:
                injected[float(amplitude)].append(float(np.mean(values[float(amplitude)])))
    return np.asarray(control), {key: np.asarray(value) for key, value in injected.items()}


def _channels(
    image: np.ndarray,
    mask: np.ndarray,
    atlas: np.ndarray,
    labels: list[int],
    injected: bool,
    amplitude: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    prep = config["preprocessing"]
    field = tapered_label_mask(atlas, labels, int(config["injection"]["taper_boundary_voxels"]))
    raw = apply_multiplicative_attenuation(image, field, amplitude) if injected else image
    built = build_channels(
        raw,
        mask,
        scaler=prep["scaler"],
        channels=list(prep["channels"]),
        clip_low=prep["clip_low"],
        clip_high=prep["clip_high"],
        rank_size=prep["rank_size"],
    )
    return built, field


def _model_features(model: torch.nn.Module, channels: np.ndarray, device: str, downsample: int) -> np.ndarray:
    tensor = torch.as_tensor(channels[None], dtype=torch.float32, device=device)
    if downsample > 1:
        tensor = F.interpolate(tensor, scale_factor=1.0 / downsample, mode="trilinear", align_corners=False)
    with torch.no_grad():
        result = model(tensor).detach().cpu().numpy()[0]
    return result.astype(np.float32)


def _aggregate_subject_features(visits: pd.DataFrame, aggregation: str) -> np.ndarray:
    ordered = visits.sort_values("age")
    matrix = np.stack(ordered["features"].to_list())
    if aggregation == "mean":
        return matrix.mean(axis=0)
    if aggregation == "annualized_rate":
        delta = float(ordered.iloc[-1]["age"] - ordered.iloc[0]["age"])
        if delta <= 0:
            raise ValueError("annualized_rate requires a positive age span")
        return (matrix[-1] - matrix[0]) / delta
    raise ValueError(f"Unsupported confirmatory aggregation: {aggregation}")


def _visit_coefficients(rows: pd.DataFrame, aggregation: str) -> list[float]:
    rows = rows.sort_values("age")
    if aggregation == "mean":
        return [1.0 / len(rows)] * len(rows)
    delta = float(rows.iloc[-1]["age"] - rows.iloc[0]["age"])
    if delta <= 0:
        raise ValueError("annualized_rate requires a positive age span")
    coefficients = [0.0] * len(rows)
    coefficients[0], coefficients[-1] = -1.0 / delta, 1.0 / delta
    return coefficients


def _cell_key(**parts: Any) -> dict[str, Any]:
    return {
        "kind": "injection",
        "cohort": parts["cohort"],
        "architecture": parts["architecture"],
        "target": parts["target"],
        "amplitude": float(parts["amplitude"]),
        "aggregation": parts["aggregation"],
        "seed": int(parts["seed"]),
        "weight_types": WEIGHT_TYPES,
    }


def run_injection_cell(
    run_root: Path,
    *,
    cohort_name: str,
    architecture: str,
    target_name: str,
    amplitude: float,
    aggregation: str,
    seed: int,
    device: str,
) -> Path:
    config = load_validation_config(run_root / "resolved_config.yaml")
    injection = config["injection"]
    if architecture not in injection["architectures"] or target_name not in injection["targets"]:
        raise ValueError("Cell is outside the preregistered architecture/target grid")
    if float(amplitude) not in [float(x) for x in injection["amplitudes"]]:
        raise ValueError("Amplitude is outside the preregistered grid")
    if aggregation not in injection["aggregations"] or int(seed) not in injection["seeds"]:
        raise ValueError("Aggregation or seed is outside the preregistered grid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; confirmatory cells do not fall back")

    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    assignment = assignment[assignment["cohort"] == cohort_name].copy()
    cohort, fs_root, parc_filename = _dataset(config, cohort_name)
    cohort["_subject_id"] = cohort["id"].astype(str)
    target = injection["targets"][target_name]
    left, right, labels = _target_labels(target)

    control_values, candidate_values = _pilot_regional_values(
        cohort, assignment, fs_root, parc_filename, labels,
        [float(x) for x in injection["amplitudes"]],
        int(injection["taper_boundary_voxels"]),
    )
    primary_amplitude, calibration = select_calibrated_amplitude(
        control_values,
        candidate_values,
        target_d=float(injection["calibration_target_cohens_d"]),
    )

    validation = assignment[assignment["split"] == "validation"]
    group_by_id = validation.set_index("subject_id")["synthetic_group"].to_dict()
    rows = cohort[cohort["_subject_id"].isin(group_by_id)].copy().sort_values(["_subject_id", "age"])
    expected_ids = set(validation["subject_id"])
    if set(rows["_subject_id"]) != expected_ids:
        missing = sorted(expected_ids - set(rows["_subject_id"]))
        raise RuntimeError(f"Missing validation subjects after cohort load: {missing}")

    prep = config["preprocessing"]
    downsample = int(prep["attribution_downsample_factor"])
    model = _build_cnn(architecture, list(prep["channels"]), seed, device)
    visit_records = []
    for _, row in rows.iterrows():
        image, mask, atlas, _ = _load_raw(row, fs_root, parc_filename)
        channels, _ = _channels(
            image, mask, atlas, labels,
            group_by_id[row["_subject_id"]] == "injected",
            float(amplitude), config,
        )
        features = _model_features(model, channels, device, downsample)
        visit_records.append({"subject_id": row["_subject_id"], "age": float(row["age"]), "features": features})
    visit_df = pd.DataFrame(visit_records)
    aggregated, groups, ordered_ids = [], [], []
    for subject_id, subject_visits in visit_df.groupby("subject_id", sort=True):
        aggregated.append(_aggregate_subject_features(subject_visits, aggregation))
        groups.append(group_by_id[subject_id])
        ordered_ids.append(subject_id)
    X = np.stack(aggregated)
    y = np.asarray(groups)
    if set(y) != {"control", "injected"}:
        raise RuntimeError("Synthetic groups are incomplete")
    weights = _compute_seed_weights(
        X, y, ("injected", "control"), WEIGHT_TYPES, seed, verbose=False,
    )
    for weight_type in WEIGHT_TYPES:
        weights[weight_type] = rank_normalize(weights[weight_type])

    sum_signed: dict[str, np.ndarray | None] = {weight: None for weight in WEIGHT_TYPES}
    sum_abs: dict[str, np.ndarray | None] = {weight: None for weight in WEIGHT_TYPES}
    metric_rows, region_rows, provenance_rows = [], [], []
    mask_sum = None
    affine = None
    lut = load_freesurfer_lut(ROOT / "FS-LUT.txt")
    for subject_id, subject_rows in rows.groupby("_subject_id", sort=True):
        subject_rows = subject_rows.sort_values("age")
        coefficients = _visit_coefficients(subject_rows, aggregation)
        subject_signed: dict[str, np.ndarray | None] = {weight: None for weight in WEIGHT_TYPES}
        subject_abs: dict[str, np.ndarray | None] = {weight: None for weight in WEIGHT_TYPES}
        subject_atlas = subject_field = None
        for coefficient, (_, row) in zip(coefficients, subject_rows.iterrows()):
            if coefficient == 0:
                continue
            image, mask, atlas, row_affine = _load_raw(row, fs_root, parc_filename)
            channels, field = _channels(
                image, mask, atlas, labels,
                group_by_id[subject_id] == "injected", float(amplitude), config,
            )
            observed, _ = _attribute_one_subject(
                model,
                torch.as_tensor(channels[None], dtype=torch.float32),
                weights,
                WEIGHT_TYPES,
                device,
                downsample,
                allow_cpu_fallback=False,
            )
            for weight_type, (signed_map, abs_map) in observed.items():
                signed_contribution = float(coefficient) * signed_map
                abs_contribution = abs(float(coefficient)) * abs_map
                subject_signed[weight_type] = signed_contribution if subject_signed[weight_type] is None else subject_signed[weight_type] + signed_contribution
                subject_abs[weight_type] = abs_contribution if subject_abs[weight_type] is None else subject_abs[weight_type] + abs_contribution
            subject_atlas, subject_field, affine = atlas, field, row_affine
            provenance_rows.append(
                {
                    "subject_id": subject_id,
                    "directory": str(row["directory"]),
                    "age": float(row["age"]),
                    "synthetic_group": group_by_id[subject_id],
                    "injected": group_by_id[subject_id] == "injected",
                    "coefficient": float(coefficient),
                    "amplitude": float(amplitude),
                }
            )
        if subject_atlas is None or subject_field is None:
            raise RuntimeError(f"No attributable visit for subject {subject_id}")
        mask_sum = subject_field if mask_sum is None else mask_sum + subject_field
        for weight_type in WEIGHT_TYPES:
            signed_map = subject_signed[weight_type]
            abs_map = subject_abs[weight_type]
            if signed_map is None or abs_map is None:
                raise RuntimeError(f"Missing map for subject={subject_id}, weight={weight_type}")
            sum_signed[weight_type] = signed_map if sum_signed[weight_type] is None else sum_signed[weight_type] + signed_map
            sum_abs[weight_type] = abs_map if sum_abs[weight_type] is None else sum_abs[weight_type] + abs_map
            metrics = localization_metrics(
                abs_map,
                subject_field,
                atlas=subject_atlas,
                left_labels=left,
                right_labels=right,
            )
            metrics.update(
                {
                    "cohort": cohort_name,
                    "architecture": architecture,
                    "target": target_name,
                    "amplitude": float(amplitude),
                    "amplitude_role": "primary" if np.isclose(amplitude, primary_amplitude) else "secondary",
                    "aggregation": aggregation,
                    "weight_type": weight_type,
                    "seed": int(seed),
                    "subject_id": subject_id,
                }
            )
            metric_rows.append(metrics)
            region_signed, region_abs = project_to_atlas(signed_map, abs_map, subject_atlas)
            for label_id, (abs_value, n_voxels) in region_abs.items():
                region_rows.append(
                    {
                        "cohort": cohort_name,
                        "architecture": architecture,
                        "target": target_name,
                        "amplitude": float(amplitude),
                        "aggregation": aggregation,
                        "weight_type": weight_type,
                        "seed": int(seed),
                        "subject_id": subject_id,
                        "label_id": label_id,
                        "region_name": lut.get(label_id, f"label_{label_id}"),
                        "n_voxels": n_voxels,
                        "signed_importance": region_signed[label_id][0],
                        "abs_importance": abs_value,
                    }
                )

    cell_dir = run_root / "cells" / cohort_name / architecture / target_name / f"amp_{amplitude:.2f}" / aggregation / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows).to_csv(cell_dir / "metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(cell_dir / "regional_stats.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(cell_dir / "provenance.csv", index=False)
    calibration.to_csv(cell_dir / "amplitude_calibration.csv", index=False)
    n_subjects = len(ordered_ids)
    if affine is None or mask_sum is None:
        raise RuntimeError("Cannot export maps without affine and injection mask")
    artifacts = ["metrics.csv", "regional_stats.csv", "provenance.csv", "amplitude_calibration.csv"]
    nib.save(nib.Nifti1Image((mask_sum / n_subjects).astype(np.float32), affine), cell_dir / "injection_mask.nii.gz")
    artifacts.append("injection_mask.nii.gz")
    for weight_type in WEIGHT_TYPES:
        signed_mean = np.asarray(sum_signed[weight_type]) / n_subjects
        abs_mean = np.asarray(sum_abs[weight_type]) / n_subjects
        for kind, image in (("signed", signed_mean), ("abs", abs_mean)):
            filename = f"voxel_map_{kind}_{weight_type}.nii.gz"
            nib.save(nib.Nifti1Image(image.astype(np.float32), affine), cell_dir / filename)
            artifacts.append(filename)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cell": _cell_key(
            cohort=cohort_name, architecture=architecture, target=target_name,
            amplitude=amplitude, aggregation=aggregation, seed=seed,
        ),
        "primary_amplitude": primary_amplitude,
        "subject_ids": ordered_ids,
        "artifacts": [
            {"path": name, "sha256": sha256_file(cell_dir / name), "size_bytes": (cell_dir / name).stat().st_size}
            for name in artifacts
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(cell_dir / "cell_manifest.json", manifest)
    return cell_dir


def run_control_cell(
    run_root: Path,
    *,
    cohort_name: str,
    architecture: str,
    seed: int,
    device: str,
) -> Path:
    """Generate unit-weight mean/constant/phase architecture-prior maps."""
    config = load_validation_config(run_root / "resolved_config.yaml")
    injection = config["injection"]
    if architecture not in injection["architectures"] or seed not in injection["seeds"]:
        raise ValueError("Control cell is outside the preregistered grid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; confirmatory cells do not fall back")
    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    validation_ids = set(
        assignment[
            (assignment["cohort"] == cohort_name)
            & (assignment["split"] == "validation")
        ]["subject_id"]
    )
    cohort, fs_root, parc_filename = _dataset(config, cohort_name)
    cohort["_subject_id"] = cohort["id"].astype(str)
    rows = cohort[cohort["_subject_id"].isin(validation_ids)].sort_values(["_subject_id", "age"])
    if set(rows["_subject_id"]) != validation_ids:
        raise RuntimeError("Control input population is missing requested validation subjects")
    prep = config["preprocessing"]
    running_sum = running_mask = None
    count = 0
    reference_atlas = affine = None
    for _, row in rows.iterrows():
        image, mask, atlas, row_affine = _load_raw(row, fs_root, parc_filename)
        channels = build_channels(
            image, mask, scaler=prep["scaler"], channels=list(prep["channels"]),
            clip_low=prep["clip_low"], clip_high=prep["clip_high"], rank_size=prep["rank_size"],
        )
        running_sum = channels.astype(np.float64) if running_sum is None else running_sum + channels
        running_mask = mask.copy() if running_mask is None else running_mask | mask
        count += 1
        if reference_atlas is None:
            reference_atlas, affine = atlas, row_affine
    if count == 0 or running_sum is None or running_mask is None or affine is None:
        raise RuntimeError("No valid control inputs")
    mean_brain = (running_sum / count).astype(np.float32)
    control_inputs = {
        "mean_brain": mean_brain,
        "constant_masked": constant_masked_input(mean_brain, running_mask),
        "phase_scrambled": phase_scramble_input(mean_brain, running_mask, seed=seed),
    }
    model = _build_cnn(architecture, list(prep["channels"]), seed, device)
    output = run_root / "controls" / cohort_name / architecture / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for input_name in config["controls"]["unit_weight_inputs"]:
        signed, absolute = unit_weight_prior(
            model,
            control_inputs[input_name],
            device=device,
            downsample_factor=int(prep["attribution_downsample_factor"]),
        )
        for kind, array in (("signed", signed), ("abs", absolute)):
            name = f"voxel_map_{kind}_unit_{input_name}.nii.gz"
            nib.save(nib.Nifti1Image(array.astype(np.float32), affine), output / name)
            artifacts.append(name)
    if reference_atlas is not None:
        lut = load_freesurfer_lut(ROOT / "FS-LUT.txt")
        region_boundary_metrics(reference_atlas, lut).to_csv(output / "region_boundary_metrics.csv", index=False)
        artifacts.append("region_boundary_metrics.csv")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cell": {
            "kind": "control",
            "cohort": cohort_name,
            "architecture": architecture,
            "seed": int(seed),
            "inputs": list(config["controls"]["unit_weight_inputs"]),
        },
        "subject_ids": sorted(validation_ids),
        "artifacts": [
            {"path": name, "sha256": sha256_file(output / name), "size_bytes": (output / name).stat().st_size}
            for name in artifacts
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "cell_manifest.json", manifest)
    return output


def collect(run_root: Path) -> None:
    """Validate every cell manifest and update the campaign manifest atomically."""
    campaign_path = run_root / "completion_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    planned = {json.dumps(cell, sort_keys=True): cell for cell in campaign["planned_cells"]}
    completed, failed = [], []
    manifest_paths = []
    for subdirectory in ("cells", "controls", "null"):
        if (run_root / subdirectory).exists():
            manifest_paths.extend((run_root / subdirectory).glob("**/*cell_manifest.json"))
    for path in sorted(manifest_paths):
        try:
            cell_manifest = json.loads(path.read_text())
            if cell_manifest.get("status") != "complete":
                raise RuntimeError("cell status is not complete")
            key = json.dumps(cell_manifest["cell"], sort_keys=True)
            if key not in planned:
                raise RuntimeError("cell is not in preregistered grid")
            for artifact in cell_manifest["artifacts"]:
                artifact_path = path.parent / artifact["path"]
                if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                    raise RuntimeError(f"missing or changed artifact {artifact_path.name}")
            completed.append(cell_manifest["cell"])
        except Exception as exc:
            failed.append({"manifest": str(path.relative_to(run_root)), "error": str(exc)})
    campaign["completed_cells"] = completed
    campaign["failed_cells"] = failed
    missing = set(planned) - {json.dumps(cell, sort_keys=True) for cell in completed}
    campaign["status"] = "complete" if not missing and not failed else "incomplete"
    campaign["missing_cell_count"] = len(missing)
    campaign["collected_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(campaign_path, campaign)
    if missing or failed:
        raise RuntimeError(f"Campaign incomplete: {len(missing)} missing, {len(failed)} invalid cells")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "attribution_validation.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--run-id", required=True)
    cell = subparsers.add_parser("injection-cell")
    cell.add_argument("--run-id", required=True)
    cell.add_argument("--cohort", choices=["nki", "ppmi"], required=True)
    cell.add_argument("--architecture", choices=["double_conv", "cov_pool"], required=True)
    cell.add_argument("--target", required=True)
    cell.add_argument("--amplitude", type=float, choices=[0.02, 0.05, 0.10], required=True)
    cell.add_argument("--aggregation", choices=["mean", "annualized_rate"], required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--device", default="cuda")
    control = subparsers.add_parser("control-cell")
    control.add_argument("--run-id", required=True)
    control.add_argument("--cohort", choices=["nki", "ppmi"], required=True)
    control.add_argument("--architecture", choices=["double_conv", "cov_pool"], required=True)
    control.add_argument("--seed", type=int, required=True)
    control.add_argument("--device", default="cuda")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            run_root = prepare(args.config, args.run_id)
            print(run_root)
            return
        config = load_validation_config(args.config)
        run_root = _resolve(config["output"]["root"]) / args.run_id
        if args.command == "injection-cell":
            output = run_injection_cell(
                run_root,
                cohort_name=args.cohort,
                architecture=args.architecture,
                target_name=args.target,
                amplitude=args.amplitude,
                aggregation=args.aggregation,
                seed=args.seed,
                device=args.device,
            )
            print(output)
        elif args.command == "control-cell":
            output = run_control_cell(
                run_root,
                cohort_name=args.cohort,
                architecture=args.architecture,
                seed=args.seed,
                device=args.device,
            )
            print(output)
        elif args.command == "collect":
            collect(run_root)
            print(f"Validated complete campaign: {run_root}")
    except Exception:
        # A traceback and nonzero exit are deliberate: scheduler wrappers must
        # never mistake missing subjects/maps or CUDA failures for completion.
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
