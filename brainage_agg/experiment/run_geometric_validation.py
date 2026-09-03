"""Cell-atomic CLI orchestrator for the geometric longitudinal validation study.

Mirrors `run_attribution_validation.py`'s philosophy: every `*-cell` command is one
atomic, SLURM-array-dispatchable unit of work that either writes a complete
`cell_manifest.json` or exits nonzero -- never a partial result. `prepare` and
`collect` are the two commands meant to run directly (cheap I/O only, no per-subject
image work); every `*-cell` command does real image work and belongs on `sbatch`, not
the login node -- matching how `scripts/submit_attribution_validation.sh` only arrays
the cell commands, never `prepare`.

`fastsurfer-cell` and `morphometry-cell` are not implemented yet. `recon-all` is out
of scope entirely as of plan §5.3's 2026-09-02 amendment (replaced by containerized
FastSurfer segmentation, `containers/fastsurfer-cpu.sif`, verified working) -- so
`fastsurfer-cell` is buildable, just not built yet. `morphometry-cell` remains
blocked: see geometric_longitudinal_validation_architecture.md §6, open question 2
(no registration library available in this environment).

`calibrate-invariance-cell` must run before any `cnn-cell`/`train-cnn-cell` with
`--variant invariant`: it fits the invariance-layer band-limit cutoff on the pilot
split only (architecture §2.2, `invariance_layer.calibrate_filter_cutoff`), using
hippocampal 5% contraction as the primary-target dose signal (plan §6.2's primary
amplitude) and the sham conditions as the noise signal to suppress. One calibration
per (cohort, architecture, seed) -- the frozen/trained backbone's own weights differ
by seed, so its nuisance-sensitivity profile does too.

Examples
--------
Prepare deterministic, disjoint pilot/validation assignments::

    python -m brainage_agg.experiment.run_geometric_validation prepare --run-id geo_v1

Calibrate the invariance-layer cutoff (real per-subject image work; run via sbatch)::

    python -m brainage_agg.experiment.run_geometric_validation calibrate-invariance-cell \\
      --run-id geo_v1 --cohort nki --architecture double_conv --seed 0 --device cuda

Train one checkpoint::

    python -m brainage_agg.experiment.run_geometric_validation train-cnn-cell \\
      --run-id geo_v1 --cohort nki --architecture double_conv --variant raw --seed 0 --device cuda

Run one CNN evaluation cell::

    python -m brainage_agg.experiment.run_geometric_validation cnn-cell \\
      --run-id geo_v1 --cohort nki --architecture double_conv --target hippocampus \\
      --seed 0 --variant raw --model-kind frozen --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.experiment.run_attribution_validation import (
    _dataset,
    _eligible_subjects,
    _load_raw,
    _target_labels,
)
from brainage_agg.geometric.config import load_geometric_config
from brainage_agg.geometric.deformation import (
    calibrate_magnitude_for_target_change,
    identity_field,
    radial_bump_field,
    rigid_subvoxel_field,
)
from brainage_agg.geometric.frozen_cnn import build_frozen_cnn, build_response_fn
from brainage_agg.geometric.invariance_layer import apply_invariance_layer, calibrate_filter_cutoff
from brainage_agg.geometric.qc import evaluate_deformation_qc
from brainage_agg.geometric.self_calibration import compute_null_response_distribution, zscore_against_null
from brainage_agg.geometric.synthesis import synthesize_followup
from brainage_agg.geometric.trained_cnn import (
    TrainingExample,
    load_trained_checkpoint,
    save_checkpoint,
    train_cnn,
)
from brainage_agg.validation.provenance import environment_snapshot, git_state, sha256_file
from scipy import ndimage as ndi
from src.preprocessing import build_channels


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def deterministic_pilot_validation_split(
    subject_ids: list[str],
    *,
    n_validation: int,
    n_pilot: int,
    seed: int,
) -> pd.DataFrame:
    """Disjoint, order-independent pilot/validation split.

    Unlike `validation/injection.py::deterministic_subject_assignment`, this carries
    no injected/control balance -- every geometric-validation subject receives the
    same set of synthetic conditions (a within-subject design), so that
    between-subject balancing concern doesn't apply here.
    """
    ids = sorted({str(subject_id) for subject_id in subject_ids})
    requested = int(n_validation) + int(n_pilot)
    if n_validation < 2 or n_pilot < 2:
        raise ValueError("Pilot and validation samples must each contain at least two subjects")
    if len(ids) < requested:
        raise ValueError(f"Requested {requested} subjects but only {len(ids)} are available")
    rng = np.random.default_rng(seed)
    selected = np.asarray(ids, dtype=object)[rng.permutation(len(ids))[:requested]]
    rows = []
    start = 0
    for split, count in (("pilot", n_pilot), ("validation", n_validation)):
        split_ids = sorted(selected[start : start + count])
        start += count
        for subject_id in split_ids:
            rows.append({"subject_id": str(subject_id), "split": split, "assignment_seed": int(seed)})
    return pd.DataFrame(rows).sort_values(["split", "subject_id"]).reset_index(drop=True)


def _planned_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    cnn = config["cnn"]
    cells: list[dict[str, Any]] = []
    for cohort in sorted(config["data"]["cohorts"]):
        for architecture in cnn["architectures"]:
            for seed in cnn["seeds"]:
                cells.append(
                    {"kind": "calibrate_invariance", "cohort": cohort, "architecture": architecture,
                     "seed": int(seed)}
                )
        for architecture in cnn["architectures"]:
            for variant in cnn["variants"]:
                for seed in cnn["seeds"]:
                    cells.append(
                        {"kind": "train", "cohort": cohort, "architecture": architecture,
                         "variant": variant, "seed": int(seed)}
                    )
        for architecture in cnn["architectures"]:
            for target in sorted(config["deformation"]["targets"]):
                for seed in cnn["seeds"]:
                    for variant in cnn["variants"]:
                        for model_kind in ("frozen", "trained"):
                            cells.append(
                                {"kind": "cnn", "cohort": cohort, "architecture": architecture,
                                 "target": target, "seed": int(seed), "variant": variant,
                                 "model_kind": model_kind}
                            )
    return cells


def target_relevant_conditions(config: dict[str, Any], target_name: str) -> dict[str, dict]:
    """The sham conditions plus every condition whose `target` is `target_name`.

    Plan §7: "Each cell processes sham and all requested amplitudes from shared
    cached inputs" -- one cell (one target) bundles its own zero-change controls
    rather than re-loading subjects for a separate sham-only cell.
    """
    conditions = config["deformation"]["conditions"]
    relevant = {
        "exact_duplicate": conditions["exact_duplicate"],
        "resampling_sham": conditions["resampling_sham"],
    }
    for name, spec in conditions.items():
        if spec.get("target") == target_name:
            relevant[name] = spec
    return relevant


def _channels_for_variant(
    image: np.ndarray,
    mask: np.ndarray,
    variant: str,
    sigma_voxels: float,
    preprocessing_config: dict[str, Any],
) -> np.ndarray:
    filtered = apply_invariance_layer(image, mask, variant=variant, sigma_voxels=sigma_voxels)
    return build_channels(
        filtered, mask,
        scaler=preprocessing_config["scaler"], channels=list(preprocessing_config["channels"]),
        clip_low=preprocessing_config["clip_low"], clip_high=preprocessing_config["clip_high"],
        rank_size=preprocessing_config["rank_size"],
    )


def _calibration_cell_dir(run_root: Path, cohort_name: str, architecture: str, seed: int) -> Path:
    return run_root / "invariance_calibration" / cohort_name / architecture / f"seed_{seed}"


def resolve_sigma_voxels(run_root: Path, *, cohort_name: str, architecture: str, seed: int, variant: str) -> float:
    """The invariance-layer cutoff to use for one (cohort, architecture, seed, variant).

    The raw variant ignores `sigma_voxels` entirely (see
    `invariance_layer.apply_invariance_layer`), so 0.0 is returned unconditionally
    without requiring a calibration cell to exist. The invariant variant requires
    `calibrate-invariance-cell` to have already run for this
    (cohort, architecture, seed); this function fails loudly rather than falling
    back to a guessed value if it hasn't.
    """
    if variant == "raw":
        return 0.0
    manifest_path = _calibration_cell_dir(run_root, cohort_name, architecture, seed) / "cell_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No invariance-filter calibration at {manifest_path}; run "
            "calibrate-invariance-cell for this (cohort, architecture, seed) first"
        )
    manifest = json.loads(manifest_path.read_text())
    return float(manifest["chosen_sigma_voxels"])


def process_condition(
    baseline_image: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    condition_name: str,
    condition_spec: dict[str, Any],
    region_mask: np.ndarray | None,
    response_fn,
    seed: int,
    magnitude_bounds: tuple[float, float],
    qc_config: dict[str, Any],
    target_sigma_voxels: float,
) -> dict[str, Any]:
    """Generate, QC, and score one condition's synthetic follow-up for one subject.

    A pure function of its arguments (no cohort I/O), so it is unit-testable with
    synthetic arrays independent of any real MRI data.
    """
    shape = baseline_image.shape
    if condition_name == "exact_duplicate":
        field = identity_field(shape)
        requested_pct = 0.0
    elif condition_name == "resampling_sham":
        rng = np.random.default_rng(seed)
        translation = rng.uniform(
            -condition_spec["max_translation_voxels"], condition_spec["max_translation_voxels"], size=3
        )
        rotation = rng.uniform(
            -condition_spec["max_rotation_degrees"], condition_spec["max_rotation_degrees"], size=3
        )
        field = rigid_subvoxel_field(
            shape, translation_voxels=tuple(translation), rotation_degrees=tuple(rotation)
        )
        requested_pct = 0.0
    else:
        if region_mask is None:
            raise ValueError(f"condition {condition_name!r} requires a target region_mask")
        _, field, _ = calibrate_magnitude_for_target_change(
            lambda magnitude: radial_bump_field(
                shape, region_mask, magnitude=magnitude,
                sigma_voxels=target_sigma_voxels, direction=condition_spec["direction"],
            ),
            region_mask,
            target_change_pct=float(condition_spec["target_change_pct"]),
            magnitude_bounds=magnitude_bounds,
        )
        requested_pct = float(condition_spec["target_change_pct"])

    qc_mask = region_mask if region_mask is not None else baseline_mask
    transition_band = ndi.binary_dilation(qc_mask, iterations=3)
    qc_result = evaluate_deformation_qc(
        field,
        region_mask=qc_mask,
        transition_band_mask=transition_band,
        requested_change_pct=requested_pct,
        tolerance_pct=float(qc_config["tolerance_pct"]),
        max_outside_change_pct=float(qc_config["max_outside_change_pct"]),
        max_boundary_gradient=float(qc_config["max_boundary_gradient"]),
        baseline_shape=shape,
        followup_shape=shape,
    )
    followup_image, _ = synthesize_followup(baseline_image, baseline_mask, field)
    response = response_fn(baseline_image, followup_image)
    return {
        "condition": condition_name,
        "requested_change_pct": requested_pct,
        "realized_change_pct": qc_result.values["realized_change_pct"],
        "qc_passed": qc_result.passed,
        "qc_checks": qc_result.checks,
        "response": response,
    }


def prepare(config_path: Path, run_id: str) -> Path:
    config = load_geometric_config(config_path)
    run_root = _resolve(config["output"]["root"]) / run_id
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_root}")
    run_root.mkdir(parents=True)

    subjects_cfg = config["subjects"]
    assignments = []
    all_subject_ids = []
    for cohort_name in sorted(config["data"]["cohorts"]):
        cohort, fs_root, parcellation_filename = _dataset(config, cohort_name)
        eligible = _eligible_subjects(cohort, fs_root, parcellation_filename)
        assigned = deterministic_pilot_validation_split(
            eligible,
            n_validation=int(subjects_cfg["validation_per_cohort"]),
            n_pilot=int(subjects_cfg["pilot_per_cohort"]),
            seed=int(subjects_cfg["assignment_seed"]),
        )
        assigned.insert(0, "cohort", cohort_name)
        assignments.append(assigned)
        all_subject_ids.extend(f"{cohort_name}:{subject_id}" for subject_id in assigned["subject_id"])

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
        "rng_seeds": {"subject_assignment": subjects_cfg["assignment_seed"]},
        "planned_cells": _planned_cells(config),
        "completed_cells": [],
        "failed_cells": [],
    }
    _atomic_json(run_root / "completion_manifest.json", manifest)
    return run_root


def calibrate_invariance_cell(
    run_root: Path,
    *,
    cohort_name: str,
    architecture: str,
    seed: int,
    device: str,
) -> Path:
    """Fit the invariance-layer band-limit cutoff on the pilot split (architecture
    §2.2). Uses hippocampal 5% contraction (plan §6.2's primary amplitude) as the
    dose signal to retain and the sham conditions as the noise signal to suppress;
    the deformation itself is computed once per pilot subject and its synthetic
    follow-up reused across every candidate sigma, matching plan §7's "same
    synthetic images are reused across methods."
    """
    config = load_geometric_config(run_root / "resolved_config.yaml")
    cnn_config = config["cnn"]
    if architecture not in cnn_config["architectures"] or seed not in cnn_config["seeds"]:
        raise ValueError("Cell is outside the preregistered calibration grid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; confirmatory cells do not fall back")

    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    pilot_ids = set(
        assignment[(assignment["cohort"] == cohort_name) & (assignment["split"] == "pilot")]["subject_id"]
    )
    cohort, fs_root, parcellation_filename = _dataset(config, cohort_name)
    cohort["_subject_id"] = cohort["id"].astype(str)
    rows = cohort[cohort["_subject_id"].isin(pilot_ids)].sort_values(["_subject_id", "age"])
    if set(rows["_subject_id"]) != pilot_ids:
        missing = sorted(pilot_ids - set(rows["_subject_id"]))
        raise RuntimeError(f"Missing pilot subjects after cohort load: {missing}")

    preprocessing_config = config["preprocessing"]
    target = config["deformation"]["targets"]["hippocampus"]
    _, _, labels = _target_labels(target)
    dose_condition = config["deformation"]["conditions"]["hippocampus_contract_05"]
    sham_condition = config["deformation"]["conditions"]["resampling_sham"]
    magnitude_bounds = tuple(float(bound) for bound in config["deformation"]["qc"]["magnitude_bounds"])
    candidate_sigmas = [float(sigma) for sigma in config["invariance"]["filter"]["cutoff_candidates"]]
    if 0.0 not in candidate_sigmas:
        raise ValueError("invariance.filter.cutoff_candidates must include 0.0 as the unfiltered reference")

    model = build_frozen_cnn(architecture, list(preprocessing_config["channels"]), seed, device)
    rng = np.random.default_rng(seed)
    sham_response: dict[float, list[float]] = {sigma: [] for sigma in candidate_sigmas}
    dose_response: dict[float, list[float]] = {sigma: [] for sigma in candidate_sigmas}

    for subject_id, subject_rows in rows.groupby("_subject_id", sort=True):
        row = subject_rows.iloc[0]
        image, mask, atlas, _ = _load_raw(row, fs_root, parcellation_filename)
        region_mask = np.isin(atlas, labels)

        _, dose_field, _ = calibrate_magnitude_for_target_change(
            lambda magnitude: radial_bump_field(
                image.shape, region_mask, magnitude=magnitude,
                sigma_voxels=float(target["sigma_voxels"]), direction=dose_condition["direction"],
            ),
            region_mask, target_change_pct=float(dose_condition["target_change_pct"]),
            magnitude_bounds=magnitude_bounds,
        )
        dose_followup, _ = synthesize_followup(image, mask, dose_field)

        translation = rng.uniform(
            -sham_condition["max_translation_voxels"], sham_condition["max_translation_voxels"], size=3
        )
        rotation = rng.uniform(
            -sham_condition["max_rotation_degrees"], sham_condition["max_rotation_degrees"], size=3
        )
        sham_field = rigid_subvoxel_field(
            image.shape, translation_voxels=tuple(translation), rotation_degrees=tuple(rotation)
        )
        sham_followup, _ = synthesize_followup(image, mask, sham_field)

        for sigma in candidate_sigmas:
            variant = "raw" if sigma == 0.0 else "invariant"
            response_fn = build_response_fn(
                model, mask, variant=variant, sigma_voxels=sigma,
                scaler=preprocessing_config["scaler"], channels=list(preprocessing_config["channels"]),
                clip_low=preprocessing_config["clip_low"], clip_high=preprocessing_config["clip_high"],
                rank_size=preprocessing_config["rank_size"], device=device,
            )
            sham_response[sigma].append(response_fn(image, sham_followup))
            dose_response[sigma].append(response_fn(image, dose_followup))

    sham_arrays = {sigma: np.array(values) for sigma, values in sham_response.items()}
    dose_arrays = {sigma: np.array(values) for sigma, values in dose_response.items()}
    calibration = calibrate_filter_cutoff(
        candidate_sigmas, sham_arrays, dose_arrays,
        reference_dose_response=dose_arrays[0.0],
        min_dose_retention_fraction=float(config["invariance"]["filter"]["min_dose_retention_fraction"]),
    )

    cell_dir = _calibration_cell_dir(run_root, cohort_name, architecture, seed)
    cell_dir.mkdir(parents=True, exist_ok=False)
    table_path = cell_dir / "calibration_table.csv"
    calibration.table.to_csv(table_path, index=False)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cell": {"kind": "calibrate_invariance", "cohort": cohort_name, "architecture": architecture, "seed": int(seed)},
        "chosen_sigma_voxels": calibration.sigma_voxels,
        "pilot_subject_ids": sorted(pilot_ids),
        "artifacts": [
            {"path": "calibration_table.csv", "sha256": sha256_file(table_path), "size_bytes": table_path.stat().st_size}
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(cell_dir / "cell_manifest.json", manifest)
    return cell_dir


def train_cnn_cell(
    run_root: Path,
    *,
    cohort_name: str,
    architecture: str,
    variant: str,
    seed: int,
    device: str,
) -> Path:
    config = load_geometric_config(run_root / "resolved_config.yaml")
    cnn_config = config["cnn"]
    if architecture not in cnn_config["architectures"] or variant not in cnn_config["variants"] or seed not in cnn_config["seeds"]:
        raise ValueError("Cell is outside the preregistered training grid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; confirmatory cells do not fall back")

    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    held_out_ids = set(assignment[assignment["cohort"] == cohort_name]["subject_id"])

    cohort, fs_root, parcellation_filename = _dataset(config, cohort_name)
    cohort["_subject_id"] = cohort["id"].astype(str)
    eligible = set(_eligible_subjects(cohort, fs_root, parcellation_filename))
    training_ids = sorted(eligible - held_out_ids)
    if len(training_ids) < 10:
        raise RuntimeError(f"Too few training subjects after excluding pilot/validation: {len(training_ids)}")

    preprocessing_config = config["preprocessing"]
    sigma_voxels = resolve_sigma_voxels(
        run_root, cohort_name=cohort_name, architecture=architecture, seed=seed, variant=variant,
    )
    rows = cohort[cohort["_subject_id"].isin(training_ids)].sort_values(["_subject_id", "age"])
    examples = []
    for subject_id, subject_rows in rows.groupby("_subject_id", sort=True):
        row = subject_rows.iloc[0]  # one cross-sectional baseline visit per training subject
        image, mask, _, _ = _load_raw(row, fs_root, parcellation_filename)
        channels = _channels_for_variant(image, mask, variant, sigma_voxels, preprocessing_config)
        examples.append(TrainingExample(subject_id=str(subject_id), channels=channels, label=float(row["age"])))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples))
    shuffled = [examples[index] for index in order]
    n_val = max(2, int(round(0.15 * len(shuffled))))
    val_examples, train_examples = shuffled[:n_val], shuffled[n_val:]

    trained_config = cnn_config["trained"]
    result = train_cnn(
        architecture, train_examples, val_examples,
        held_out_subject_ids=held_out_ids, label_name=trained_config["label"], seed=seed,
        max_epochs=int(trained_config["max_epochs"]), patience=int(trained_config["patience"]),
        learning_rate=float(trained_config["learning_rate"]), weight_decay=float(trained_config["weight_decay"]),
        batch_size=int(trained_config["batch_size"]), device=device,
    )
    checkpoint_dir = _resolve(trained_config["checkpoint_dir"]) / run_root.name / cohort_name / variant
    checkpoint_path = save_checkpoint(result, checkpoint_dir)

    cell_dir = run_root / "training" / cohort_name / architecture / variant / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=False)
    history_path = cell_dir / "training_history.csv"
    pd.DataFrame([vars(epoch) for epoch in result.history]).to_csv(history_path, index=False)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cell": {"kind": "train", "cohort": cohort_name, "architecture": architecture, "variant": variant, "seed": int(seed)},
        "checkpoint_path": str(checkpoint_path),
        "n_training_subjects": len(train_examples),
        "n_val_subjects": len(val_examples),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "artifacts": [
            {"path": "training_history.csv", "sha256": sha256_file(history_path), "size_bytes": history_path.stat().st_size}
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(cell_dir / "cell_manifest.json", manifest)
    return cell_dir


def cnn_cell(
    run_root: Path,
    *,
    cohort_name: str,
    architecture: str,
    target_name: str,
    seed: int,
    variant: str,
    model_kind: str,
    device: str,
) -> Path:
    config = load_geometric_config(run_root / "resolved_config.yaml")
    cnn_config = config["cnn"]
    if (
        architecture not in cnn_config["architectures"]
        or variant not in cnn_config["variants"]
        or seed not in cnn_config["seeds"]
        or target_name not in config["deformation"]["targets"]
        or model_kind not in ("frozen", "trained")
    ):
        raise ValueError("Cell is outside the preregistered evaluation grid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; confirmatory cells do not fall back")

    assignment = pd.read_csv(run_root / "subject_assignment.csv", dtype={"subject_id": str})
    validation_ids = set(
        assignment[(assignment["cohort"] == cohort_name) & (assignment["split"] == "validation")]["subject_id"]
    )
    cohort, fs_root, parcellation_filename = _dataset(config, cohort_name)
    cohort["_subject_id"] = cohort["id"].astype(str)
    rows = cohort[cohort["_subject_id"].isin(validation_ids)].sort_values(["_subject_id", "age"])
    if set(rows["_subject_id"]) != validation_ids:
        missing = sorted(validation_ids - set(rows["_subject_id"]))
        raise RuntimeError(f"Missing validation subjects after cohort load: {missing}")

    target = config["deformation"]["targets"][target_name]
    left_labels, right_labels, labels = _target_labels(target)
    preprocessing_config = config["preprocessing"]
    sigma_voxels = resolve_sigma_voxels(
        run_root, cohort_name=cohort_name, architecture=architecture, seed=seed, variant=variant,
    )
    magnitude_bounds = tuple(float(bound) for bound in config["deformation"]["qc"]["magnitude_bounds"])
    qc_config = config["deformation"]["qc"]

    channels_in = len(preprocessing_config["channels"])
    if model_kind == "frozen":
        model = build_frozen_cnn(architecture, list(preprocessing_config["channels"]), seed, device)
    else:
        checkpoint_dir = _resolve(cnn_config["trained"]["checkpoint_dir"]) / run_root.name / cohort_name / variant
        checkpoint_path = checkpoint_dir / f"{architecture}__{cnn_config['trained']['label']}__seed_{seed}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint at {checkpoint_path}; run train-cnn-cell for this "
                "(cohort, architecture, variant, seed) first"
            )
        model = load_trained_checkpoint(checkpoint_path, device=device)
    del channels_in  # only used to document the frozen/trained symmetry above

    conditions = target_relevant_conditions(config, target_name)
    rows_out: list[dict[str, Any]] = []
    for subject_id, subject_rows in rows.groupby("_subject_id", sort=True):
        row = subject_rows.iloc[0]
        image, mask, atlas, _ = _load_raw(row, fs_root, parcellation_filename)
        region_mask = np.isin(atlas, labels)
        response_fn = build_response_fn(
            model, mask, variant=variant, sigma_voxels=sigma_voxels,
            scaler=preprocessing_config["scaler"], channels=list(preprocessing_config["channels"]),
            clip_low=preprocessing_config["clip_low"], clip_high=preprocessing_config["clip_high"],
            rank_size=preprocessing_config["rank_size"], device=device,
        )
        condition_results = {}
        for condition_name, condition_spec in conditions.items():
            condition_results[condition_name] = process_condition(
                image, mask,
                condition_name=condition_name, condition_spec=condition_spec,
                region_mask=None if condition_name in ("exact_duplicate", "resampling_sham") else region_mask,
                response_fn=response_fn, seed=seed, magnitude_bounds=magnitude_bounds,
                qc_config=qc_config, target_sigma_voxels=float(target["sigma_voxels"]),
            )
        # One shared null distribution per subject; every condition's response is
        # z-scored against it (see self_calibration.compute_null_response_distribution
        # docstring) rather than each condition getting its own independent null draw.
        null_responses = compute_null_response_distribution(
            image, mask, response_fn,
            n_resamples=int(config["invariance"]["self_calibration"]["k_null_resamples"]),
            seed=seed,
            max_translation_voxels=float(config["invariance"]["self_calibration"]["max_translation_voxels"]),
            max_rotation_degrees=float(config["invariance"]["self_calibration"]["max_rotation_degrees"]),
        )
        for condition_name, result in condition_results.items():
            scored = zscore_against_null(result["response"], null_responses)
            rows_out.append(
                {
                    "cohort": cohort_name, "architecture": architecture, "target": target_name,
                    "seed": int(seed), "variant": variant, "model_kind": model_kind,
                    "subject_id": str(subject_id), "condition": condition_name,
                    "requested_change_pct": result["requested_change_pct"],
                    "realized_change_pct": result["realized_change_pct"],
                    "qc_passed": result["qc_passed"], "response": result["response"],
                    "self_calibration_z_score": scored["z_score"],
                    "self_calibration_null_mean": scored["null_mean"],
                    "self_calibration_null_std": scored["null_std"],
                }
            )

    cell_dir = run_root / "cells" / cohort_name / architecture / target_name / f"seed_{seed}" / variant / model_kind
    cell_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = cell_dir / "metrics.csv"
    pd.DataFrame(rows_out).to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cell": {
            "kind": "cnn", "cohort": cohort_name, "architecture": architecture, "target": target_name,
            "seed": int(seed), "variant": variant, "model_kind": model_kind,
        },
        "subject_ids": sorted(validation_ids),
        "artifacts": [
            {"path": "metrics.csv", "sha256": sha256_file(metrics_path), "size_bytes": metrics_path.stat().st_size}
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(cell_dir / "cell_manifest.json", manifest)
    return cell_dir


def collect(run_root: Path) -> None:
    """Validate every cell manifest and update the campaign manifest atomically."""
    campaign_path = run_root / "completion_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    planned = {json.dumps(cell, sort_keys=True): cell for cell in campaign["planned_cells"]}
    completed, failed = [], []
    manifest_paths = []
    for subdirectory in ("cells", "training", "invariance_calibration"):
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "geometric_longitudinal_validation.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--run-id", required=True)

    calibrate = subparsers.add_parser("calibrate-invariance-cell")
    calibrate.add_argument("--run-id", required=True)
    calibrate.add_argument("--cohort", choices=["nki", "ppmi"], required=True)
    calibrate.add_argument("--architecture", choices=["double_conv", "cov_pool"], required=True)
    calibrate.add_argument("--seed", type=int, required=True)
    calibrate.add_argument("--device", default="cuda")

    train = subparsers.add_parser("train-cnn-cell")
    train.add_argument("--run-id", required=True)
    train.add_argument("--cohort", choices=["nki", "ppmi"], required=True)
    train.add_argument("--architecture", choices=["double_conv", "cov_pool"], required=True)
    train.add_argument("--variant", choices=["raw", "invariant"], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda")

    cnn = subparsers.add_parser("cnn-cell")
    cnn.add_argument("--run-id", required=True)
    cnn.add_argument("--cohort", choices=["nki", "ppmi"], required=True)
    cnn.add_argument("--architecture", choices=["double_conv", "cov_pool"], required=True)
    cnn.add_argument("--target", required=True)
    cnn.add_argument("--seed", type=int, required=True)
    cnn.add_argument("--variant", choices=["raw", "invariant"], required=True)
    cnn.add_argument("--model-kind", choices=["frozen", "trained"], required=True)
    cnn.add_argument("--device", default="cuda")

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
        config = load_geometric_config(args.config)
        run_root = _resolve(config["output"]["root"]) / args.run_id
        if args.command == "calibrate-invariance-cell":
            output = calibrate_invariance_cell(
                run_root, cohort_name=args.cohort, architecture=args.architecture,
                seed=args.seed, device=args.device,
            )
            print(output)
        elif args.command == "train-cnn-cell":
            output = train_cnn_cell(
                run_root, cohort_name=args.cohort, architecture=args.architecture,
                variant=args.variant, seed=args.seed, device=args.device,
            )
            print(output)
        elif args.command == "cnn-cell":
            output = cnn_cell(
                run_root, cohort_name=args.cohort, architecture=args.architecture,
                target_name=args.target, seed=args.seed, variant=args.variant,
                model_kind=args.model_kind, device=args.device,
            )
            print(output)
        elif args.command == "collect":
            collect(run_root)
            print(f"Validated complete campaign: {run_root}")
    except Exception:
        # A traceback and nonzero exit are deliberate: scheduler wrappers must
        # never mistake a missing checkpoint, missing subjects, or CUDA failure
        # for completion.
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
