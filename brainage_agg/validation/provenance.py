"""Resolved-configuration, environment, and strict completion manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> dict[str, Any]:
    """Return commit and dirty state without mutating the repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, text=True,
            capture_output=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "nibabel", "shap", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = None
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "version": torch.version.cuda}
    except ImportError:
        pass
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": packages,
        "cuda": cuda,
    }


def initialize_run(
    config_path: Path,
    run_dir: Path,
    *,
    project_root: Path,
    run_id: str,
    subject_ids: list[str],
    rng_seeds: dict[str, Any],
    planned_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the immutable run contract before computation begins."""
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (run_dir / "subject_ids.json").write_text(json.dumps(sorted(map(str, subject_ids)), indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root),
        "config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "subject_ids": sorted(map(str, subject_ids)),
        "rng_seeds": rng_seeds,
        "planned_cells": planned_cells,
        "completed_cells": [],
        "artifacts": [],
        "errors": [],
    }
    _write_manifest(run_dir, manifest)
    return manifest


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    temporary = run_dir / "completion_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(run_dir / "completion_manifest.json")


def record_cell(
    run_dir: Path,
    manifest: dict[str, Any],
    cell: dict[str, Any],
    artifacts: list[Path],
) -> None:
    records = []
    for path in artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"Expected artifact was not written: {path}")
        records.append(
            {
                "path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest["completed_cells"].append(cell)
    manifest["artifacts"].extend(records)
    _write_manifest(run_dir, manifest)


def finalize_run(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Mark complete only when planned/completed cells match exactly."""
    planned = {json.dumps(cell, sort_keys=True) for cell in manifest["planned_cells"]}
    completed = {json.dumps(cell, sort_keys=True) for cell in manifest["completed_cells"]}
    missing, unexpected = planned - completed, completed - planned
    if missing or unexpected:
        raise RuntimeError(
            f"Cannot complete run: {len(missing)} missing and {len(unexpected)} unexpected cells"
        )
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(run_dir, manifest)


def validate_completed_manifest(run_dir: Path, required_files: list[str] | None = None) -> dict[str, Any]:
    path = run_dir / "completion_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing completion manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {run_dir}")
    planned = {json.dumps(cell, sort_keys=True) for cell in manifest["planned_cells"]}
    completed = {json.dumps(cell, sort_keys=True) for cell in manifest["completed_cells"]}
    if planned != completed:
        raise RuntimeError("Manifest planned/completed cell sets differ")
    for artifact in manifest.get("artifacts", []):
        artifact_path = run_dir / artifact["path"]
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"Missing or changed artifact: {artifact_path}")
    for relative in required_files or []:
        if not (run_dir / relative).is_file():
            raise FileNotFoundError(f"Required artifact missing: {relative}")
    return manifest
