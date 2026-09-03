"""Loading and validation of the preregistered experiment contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


LOCKED = {
    "cohorts": {"nki", "ppmi"},
    "architectures": ["double_conv", "cov_pool"],
    "aggregations": ["mean", "annualized_rate"],
    "weight_types": ["tstat", "shap", "hybrid"],
    "amplitudes": [0.02, 0.05, 0.10],
    "null_architecture": "double_conv",
    "null_permutations": 199,
    "null_seeds": [0, 1, 2, 3, 4],
    "null_subjects": 50,
    "max_concurrent_gpu_tasks": 2,
}


def load_validation_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    validate_locked_config(config)
    return config


def validate_locked_config(config: dict[str, Any]) -> None:
    """Reject changes to confirmatory factors that would invalidate preregistration."""
    errors = []
    injection = config.get("injection", {})
    null = config.get("cohort_null", {})
    if set(config.get("data", {}).get("cohorts", {})) != LOCKED["cohorts"]:
        errors.append("data.cohorts must contain exactly nki and ppmi")
    for key in ("architectures", "aggregations", "weight_types", "amplitudes"):
        actual = injection.get(key)
        expected = LOCKED[key]
        if actual != expected:
            errors.append(f"injection.{key} must remain {expected!r}, got {actual!r}")
    checks = {
        "architecture": "null_architecture",
        "n_permutations": "null_permutations",
        "seeds": "null_seeds",
        "n_subjects": "null_subjects",
        "max_concurrent_gpu_tasks": "max_concurrent_gpu_tasks",
    }
    for config_key, locked_key in checks.items():
        if null.get(config_key) != LOCKED[locked_key]:
            errors.append(f"cohort_null.{config_key} must remain {LOCKED[locked_key]!r}")
    if errors:
        raise ValueError("Invalid preregistered configuration:\n- " + "\n- ".join(errors))
