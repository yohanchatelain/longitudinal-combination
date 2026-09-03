"""Loading and validation of the preregistered geometric-validation config contract.

Mirrors `validation/config.py`'s locked-schema discipline (reject any load that
drifts from the preregistered confirmatory factors), applied to a separate config and
run identifier so this study can never be pooled with the intensity-injection study
(plan §10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

LOCKED = {
    "cohorts": {"nki", "ppmi"},
    "architectures": ["double_conv", "cov_pool"],
    "variants": ["raw", "invariant"],
    "seeds": [0, 1, 2, 3, 4],
    "targets": {"hippocampus", "cerebellar_white_matter"},
    "k_null_resamples": 5,
    "calibration_fpr_max": 0.05,
    "calibration_upper_ci_max": 0.10,
    "superiority_min_improvement_pct": 20.0,
    "invariant_max_auroc_reduction": 0.05,
}


def load_geometric_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    validate_locked_config(config)
    return config


def validate_locked_config(config: dict[str, Any]) -> None:
    """Reject changes to confirmatory factors that would invalidate preregistration."""
    errors = []
    cnn = config.get("cnn", {})
    deformation = config.get("deformation", {})
    invariance = config.get("invariance", {})
    gates = config.get("decision_gates", {})

    if set(config.get("data", {}).get("cohorts", {})) != LOCKED["cohorts"]:
        errors.append("data.cohorts must contain exactly nki and ppmi")
    for key in ("architectures", "variants", "seeds"):
        actual = cnn.get(key)
        expected = LOCKED[key]
        if actual != expected:
            errors.append(f"cnn.{key} must remain {expected!r}, got {actual!r}")
    if set(deformation.get("targets", {})) != LOCKED["targets"]:
        errors.append(f"deformation.targets must contain exactly {LOCKED['targets']!r}")

    self_calibration = invariance.get("self_calibration", {})
    if self_calibration.get("k_null_resamples") != LOCKED["k_null_resamples"]:
        errors.append(
            f"invariance.self_calibration.k_null_resamples must remain {LOCKED['k_null_resamples']!r}"
        )

    calibration_gate = gates.get("calibration", {})
    if calibration_gate.get("fpr_max") != LOCKED["calibration_fpr_max"]:
        errors.append(f"decision_gates.calibration.fpr_max must remain {LOCKED['calibration_fpr_max']!r}")
    if calibration_gate.get("upper_ci_max") != LOCKED["calibration_upper_ci_max"]:
        errors.append(
            f"decision_gates.calibration.upper_ci_max must remain {LOCKED['calibration_upper_ci_max']!r}"
        )

    superiority_gate = gates.get("superiority", {})
    if superiority_gate.get("min_improvement_pct") != LOCKED["superiority_min_improvement_pct"]:
        errors.append(
            "decision_gates.superiority.min_improvement_pct must remain "
            f"{LOCKED['superiority_min_improvement_pct']!r}"
        )

    invariant_gate = gates.get("invariant_vs_raw", {})
    if invariant_gate.get("max_auroc_reduction") != LOCKED["invariant_max_auroc_reduction"]:
        errors.append(
            "decision_gates.invariant_vs_raw.max_auroc_reduction must remain "
            f"{LOCKED['invariant_max_auroc_reduction']!r}"
        )

    if errors:
        raise ValueError("Invalid preregistered configuration:\n- " + "\n- ".join(errors))
