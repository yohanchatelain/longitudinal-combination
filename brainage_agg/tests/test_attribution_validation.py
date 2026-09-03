"""CPU-only tests for the preregistered voxel-attribution validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import nibabel as nib

from brainage_agg.analysis.importance_weights import compute_w_t
from brainage_agg.analysis.voxel_attribution import compute_attribution_map
from brainage_agg.validation.config import load_validation_config, validate_locked_config
from brainage_agg.validation.gates import evaluate_decision_gates
from brainage_agg.validation.injection import (
    apply_multiplicative_attenuation,
    deterministic_subject_assignment,
    select_calibrated_amplitude,
    tapered_label_mask,
)
from brainage_agg.validation.metrics import (
    boundary_bias_regression,
    localization_metrics,
    region_boundary_metrics,
)
from brainage_agg.validation.provenance import (
    finalize_run,
    initialize_run,
    record_cell,
    validate_completed_manifest,
)
from brainage_agg.validation.statistics import empirical_region_pvalues, maxT_pseudo_null_calibration
from brainage_agg.analysis.report_attribution_validation import build_report
from brainage_agg.validation.provenance import sha256_file
from brainage_agg.experiment.run_attribution_validation import _planned_cells

ROOT = Path(__file__).resolve().parents[2]


def test_preregistered_config_is_locked():
    config = load_validation_config(ROOT / "configs" / "attribution_validation.yaml")
    assert config["cohort_null"]["n_permutations"] == 199
    changed = json.loads(json.dumps(config))
    changed["injection"]["amplitudes"] = [0.1]
    with pytest.raises(ValueError, match="amplitudes"):
        validate_locked_config(changed)
    cells = _planned_cells(config)
    assert sum(cell["kind"] == "injection" for cell in cells) == 960
    assert sum(cell["kind"] == "control" for cell in cells) == 40
    assert sum(cell["kind"] == "null" for cell in cells) == 4


def test_deterministic_assignment_is_order_independent_and_disjoint():
    ids = [f"sub-{index:03d}" for index in range(20)]
    first = deterministic_subject_assignment(ids, n_validation=10, n_pilot=6, seed=12)
    second = deterministic_subject_assignment(reversed(ids), n_validation=10, n_pilot=6, seed=12)
    pd.testing.assert_frame_equal(first, second)
    pilot = set(first[first["split"] == "pilot"]["subject_id"])
    validation = set(first[first["split"] == "validation"]["subject_id"])
    assert pilot.isdisjoint(validation)
    for _, split in first.groupby("split"):
        assert abs((split["synthetic_group"] == "injected").sum() - (split["synthetic_group"] == "control").sum()) <= 1


def test_two_voxel_taper_and_raw_attenuation():
    atlas = np.zeros((9, 9, 9), dtype=np.int16)
    atlas[1:8, 1:8, 1:8] = 12
    field = tapered_label_mask(atlas, [12], boundary_voxels=2)
    assert np.isclose(field[1, 4, 4], 1 / 3)
    assert np.isclose(field[2, 4, 4], 2 / 3)
    assert field[4, 4, 4] == 1.0
    assert field[0, 4, 4] == 0.0
    volume = np.full(atlas.shape, 100.0, dtype=np.float32)
    attenuated = apply_multiplicative_attenuation(volume, field, 0.10)
    assert attenuated[4, 4, 4] == 90.0
    assert attenuated[0, 4, 4] == 100.0


def test_calibration_selects_nearest_d_and_breaks_tie_downward():
    control = np.array([0.0, 1.0, 2.0, 3.0])
    injected = {
        0.02: np.array([0.1, 1.1, 2.1, 3.1]),
        0.05: np.array([1.0, 2.0, 3.0, 4.0]),
        0.10: np.array([2.0, 3.0, 4.0, 5.0]),
    }
    amplitude, table = select_calibrated_amplitude(control, injected, target_d=0.8)
    assert amplitude == 0.05
    assert table["amplitude"].tolist() == [0.02, 0.05, 0.10]


def test_empirical_pvalues_and_maxT_are_exact():
    observed = np.array([4.0, 2.0])
    permutations = np.array([[1.0, 0.0], [3.0, 3.0], [0.0, 1.0]])
    pointwise, adjusted = empirical_region_pvalues(observed, permutations)
    np.testing.assert_allclose(pointwise, [0.25, 0.5])
    np.testing.assert_allclose(adjusted, [0.25, 0.5])
    assert np.all(adjusted >= pointwise)


def test_leave_one_permutation_out_null_calibration():
    permutations = np.zeros((19, 4), dtype=float)
    result = maxT_pseudo_null_calibration(permutations, alpha=0.05)
    assert result["n_pseudo_null"] == 19
    assert result["maxT_fwer"] == 0.0
    assert 0.0 < result["maxT_fwer_ci_upper"] < 0.2


def test_boundary_metrics_use_geometry_volume_and_tissue():
    atlas = np.zeros((8, 8, 8), dtype=np.int16)
    atlas[1:6, 1:6, 1:6] = 2
    atlas[6:8, 6:8, 6:8] = 1028
    frame = region_boundary_metrics(atlas, {2: "Left-Cerebral-White-Matter", 1028: "ctx-lh-superiorfrontal"})
    assert set(["boundary_voxel_fraction", "log_volume", "tissue_class", "region_volume_quintile"]) <= set(frame)
    white = frame.set_index("label_id").loc[2]
    cortical = frame.set_index("label_id").loc[1028]
    assert white["tissue_class"] == "white_matter"
    assert cortical["tissue_class"] == "cortical_gray"
    assert white["boundary_voxel_fraction"] < cortical["boundary_voxel_fraction"]


def test_boundary_bias_regression_uses_all_predeclared_covariates():
    boundary = pd.DataFrame(
        {
            "label_id": np.arange(10),
            "boundary_voxel_fraction": np.linspace(0.1, 0.9, 10),
            "log_volume": np.log(np.arange(10) + 2),
            "tissue_class": ["gray", "white"] * 5,
        }
    )
    importance = pd.DataFrame(
        {
            "label_id": np.arange(10),
            "abs_importance": 2 * boundary["boundary_voxel_fraction"] - boundary["log_volume"] + ([0.3, 0.0] * 5),
        }
    )
    coefficients = boundary_bias_regression(importance, boundary)
    assert {"boundary_voxel_fraction", "log_volume", "tissue_white"} <= set(coefficients["term"])
    assert coefficients["r_squared"].iloc[0] > 0.99


class _IdentityVoxels(torch.nn.Module):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self.out_features = int(np.prod(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


def test_tiny_linear_model_localizes_injection_but_random_labels_do_not():
    rng = np.random.default_rng(7)
    shape = (8, 8, 8)
    target = np.zeros(shape, dtype=bool)
    target[2:6, 2:6, 2:6] = True
    n = 200
    volumes = rng.normal(size=(n, *shape)).astype(np.float32)
    labels = np.array(["injected"] * (n // 2) + ["control"] * (n // 2))
    volumes[: n // 2, target] -= 1.0
    features = volumes.reshape(n, -1)
    weights = compute_w_t(features, labels, ("injected", "control"))
    model = _IdentityVoxels(shape)
    _, attribution = compute_attribution_map(
        model,
        torch.as_tensor(volumes[:1, None]),
        weights,
        device="cpu",
        downsample_factor=1,
    )
    localized = localization_metrics(attribution, target)
    assert localized["voxel_auroc"] > 0.9
    assert localized["normalized_auprc_lift"] > 5

    random_labels = labels[rng.permutation(n)]
    random_weights = compute_w_t(features, random_labels, ("injected", "control"))
    _, random_map = compute_attribution_map(
        model,
        torch.as_tensor(volumes[:1, None]),
        random_weights,
        device="cpu",
        downsample_factor=1,
    )
    random_metric = localization_metrics(random_map, target)
    assert random_metric["voxel_auroc"] < 0.65


def test_decision_gate_policy_general_scoped_and_diagnostic():
    config = load_validation_config(ROOT / "configs" / "attribution_validation.yaml")
    localization_rows = []
    null_rows = []
    specificity_rows = []
    for cohort in ("nki", "ppmi"):
        null_rows.append({"cohort": cohort, "architecture": "double_conv", "maxT_fwer": 0.03, "maxT_fwer_ci_upper": 0.08})
        for architecture in ("double_conv", "cov_pool"):
            localization_rows.append(
                {
                    "cohort": cohort, "architecture": architecture, "target": "putamen",
                    "amplitude_role": "primary", "voxel_auroc": 0.8,
                    "voxel_auroc_ci_lower": 0.6, "normalized_auprc_lift": 2.5,
                    "normalized_auprc_lift_ci_lower": 1.2, "bilateral_top5_hit_rate": 0.9,
                }
            )
            for comparison in ("within_minus_between", "within_minus_prior"):
                specificity_rows.append({"cohort": cohort, "architecture": architecture, "target": "putamen", "comparison": comparison, "ci_lower": 0.1})
    decision = evaluate_decision_gates(
        pd.DataFrame(localization_rows), pd.DataFrame(null_rows), pd.DataFrame(specificity_rows), config["decision_gates"]
    )
    assert decision["claim_level"] == "general"
    failed = pd.DataFrame(localization_rows)
    failed.loc[0, "voxel_auroc"] = 0.4
    decision = evaluate_decision_gates(failed, pd.DataFrame(null_rows), pd.DataFrame(specificity_rows), config["decision_gates"])
    assert decision["claim_level"] in {"scoped", "diagnostic_only"}
    assert decision["roi_concordance_used_as_gate"] is False


def test_cpu_integration_manifest_schema_and_failure_propagation(tmp_path: Path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text("schema_version: 1\n")
    run_dir = tmp_path / "run"
    cells = [{"seed": seed, "n_permutations": 3} for seed in (0, 1)]
    manifest = initialize_run(
        config_path,
        run_dir,
        project_root=ROOT,
        run_id="tiny",
        subject_ids=["s1", "s2"],
        rng_seeds={"cnn": [0, 1], "permutation": [10, 11, 12]},
        planned_cells=cells,
    )
    for cell in cells:
        path = run_dir / f"seed_{cell['seed']}.npz"
        np.savez_compressed(
            path,
            observed=np.ones(2),
            permutations=np.zeros((3, 2)),
            subject_ids=np.array(["s1", "s2"]),
        )
        record_cell(run_dir, manifest, cell, [path])
    finalize_run(run_dir, manifest)
    validated = validate_completed_manifest(run_dir)
    assert validated["status"] == "complete"
    assert len(validated["subject_ids"]) == 2
    assert len(validated["completed_cells"]) == 2

    broken_dir = tmp_path / "broken"
    broken = initialize_run(
        config_path,
        broken_dir,
        project_root=ROOT,
        run_id="broken",
        subject_ids=["s1", "s2"],
        rng_seeds={"cnn": [0, 1]},
        planned_cells=cells,
    )
    artifact = broken_dir / "only_one.npz"
    np.savez_compressed(artifact, permutations=np.zeros((3, 2)))
    record_cell(broken_dir, broken, cells[0], [artifact])
    with pytest.raises(RuntimeError, match="missing"):
        finalize_run(broken_dir, broken)


def test_unified_report_writes_complete_paper_package(tmp_path: Path):
    run_dir = tmp_path / "campaign"
    cells_root = run_dir / "cells"
    planned = []
    rng = np.random.default_rng(4)
    bases = {
        "putamen": np.pad(np.ones((2, 2, 2)), 1),
        "hippocampus": np.pad(np.eye(2)[:, :, None] * np.ones((1, 1, 2)), 1),
    }
    for cohort in ("nki", "ppmi"):
        for architecture in ("double_conv", "cov_pool"):
            for target in ("putamen", "hippocampus"):
                for seed in (0, 1):
                    cell = {
                        "kind": "injection", "cohort": cohort, "architecture": architecture,
                        "target": target, "amplitude": 0.05, "aggregation": "mean",
                        "seed": seed, "weight_types": ["tstat"],
                    }
                    planned.append(cell)
                    directory = cells_root / cohort / architecture / target / "amp_0.05" / "mean" / f"seed_{seed}"
                    directory.mkdir(parents=True)
                    metrics = pd.DataFrame(
                        {
                            "cohort": [cohort, cohort], "architecture": [architecture] * 2,
                            "target": [target] * 2, "amplitude": [0.05] * 2,
                            "amplitude_role": ["primary"] * 2, "aggregation": ["mean"] * 2,
                            "weight_type": ["tstat"] * 2, "seed": [seed] * 2,
                            "subject_id": ["s1", "s2"], "voxel_auroc": [0.8, 0.82],
                            "normalized_auprc_lift": [2.5, 2.6],
                            "attribution_mass_fraction": [0.4, 0.45],
                            "bilateral_top5_hit": [1.0, 1.0],
                        }
                    )
                    metrics.to_csv(directory / "metrics.csv", index=False)
                    image = bases[target] + rng.normal(0, 0.001, size=(4, 4, 4))
                    nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), directory / "voxel_map_abs_tstat.nii.gz")
                    artifacts = []
                    for name in ("metrics.csv", "voxel_map_abs_tstat.nii.gz"):
                        artifacts.append({"path": name, "sha256": sha256_file(directory / name), "size_bytes": (directory / name).stat().st_size})
                    (directory / "cell_manifest.json").write_text(
                        json.dumps({"status": "complete", "cell": cell, "primary_amplitude": 0.05, "artifacts": artifacts})
                    )
            for seed in (0, 1):
                control = run_dir / "controls" / cohort / architecture / f"seed_{seed}"
                control.mkdir(parents=True)
                prior = rng.normal(size=(4, 4, 4)).astype(np.float32)
                nib.save(nib.Nifti1Image(prior, np.eye(4)), control / "voxel_map_abs_unit_mean_brain.nii.gz")
    (run_dir / "completion_manifest.json").write_text(
        json.dumps({"status": "complete", "planned_cells": planned, "completed_cells": planned})
    )
    null_paths = []
    for cohort in ("nki", "ppmi"):
        directory = tmp_path / cohort
        directory.mkdir()
        path = directory / "permutation_region_stats_double_conv__mean__tstat__tiny.npz"
        np.savez_compressed(
            path,
            label_ids=np.array([1, 2]),
            observed_abs=np.array([1.0, 0.5]),
            permutation_abs=np.zeros((5, 19, 2), dtype=np.float32),
        )
        null_paths.append(path)
    output = tmp_path / "paper"
    decision = build_report(
        [run_dir], output,
        config_path=ROOT / "configs" / "attribution_validation.yaml",
        null_paths=null_paths,
        roi_paths=[],
    )
    expected = {
        "validation_summary.csv", "permutation_region_stats.npz", "decision.json",
        "figure_injected_localization.png", "figure_null_calibration.png",
        "figure_target_specificity.png", "figure_roi_concordance.png",
        "manuscript_attribution_validation.md",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert decision["roi_concordance_used_as_gate"] is False
