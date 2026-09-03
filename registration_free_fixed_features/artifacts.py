from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "nibabel", "scikit-learn", "PyYAML", "SimpleITK"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "RFFF_N4_THREADS",
            )
        },
    }


def git_state(repository: str | Path) -> dict[str, Any]:
    root = Path(repository)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=destination.name, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


@dataclass(frozen=True)
class VisitFeatureArtifact:
    X: np.ndarray
    feature_names: tuple[str, ...]
    subject_ids: np.ndarray
    visit_ids: np.ndarray
    elapsed_years: np.ndarray
    image_paths: np.ndarray
    image_hashes: np.ndarray
    mask_hashes: np.ndarray
    qc_json: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        n_rows = len(self.X)
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise ValueError("Feature matrix and schema are not aligned")
        if not np.isfinite(self.X).all():
            raise ValueError("Feature matrix contains non-finite values")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names are not unique")
        arrays: Sequence[np.ndarray] = (
            self.subject_ids,
            self.visit_ids,
            self.elapsed_years,
            self.image_paths,
            self.image_hashes,
            self.mask_hashes,
            self.qc_json,
        )
        if any(len(array) != n_rows for array in arrays):
            raise ValueError("Visit metadata arrays must match feature row count")
        for subject in np.unique(self.subject_ids):
            times = self.elapsed_years[self.subject_ids == subject]
            if len(times) != len(np.unique(times)):
                raise ValueError(f"Subject {subject} has duplicate elapsed times")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, prefix=destination.name, delete=False
        ) as handle:
            np.savez_compressed(
                handle,
                X=np.asarray(self.X, dtype=np.float32),
                feature_names=np.asarray(self.feature_names, dtype=str),
                subject_ids=np.asarray(self.subject_ids, dtype=str),
                visit_ids=np.asarray(self.visit_ids, dtype=str),
                elapsed_years=np.asarray(self.elapsed_years, dtype=np.float32),
                image_paths=np.asarray(self.image_paths, dtype=str),
                image_hashes=np.asarray(self.image_hashes, dtype=str),
                mask_hashes=np.asarray(self.mask_hashes, dtype=str),
                qc_json=np.asarray(self.qc_json, dtype=str),
                metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            )
            temporary = Path(handle.name)
        temporary.replace(destination)
        write_json_atomic(
            destination.with_suffix(destination.suffix + ".json"),
            {
                "artifact_sha256": sha256_file(destination),
                "n_rows": int(self.X.shape[0]),
                "n_features": int(self.X.shape[1]),
                "feature_names": list(self.feature_names),
                "metadata": self.metadata,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "VisitFeatureArtifact":
        with np.load(path, allow_pickle=False) as data:
            artifact = cls(
                X=np.asarray(data["X"], dtype=np.float32),
                feature_names=tuple(str(value) for value in data["feature_names"]),
                subject_ids=np.asarray(data["subject_ids"]).astype(str),
                visit_ids=np.asarray(data["visit_ids"]).astype(str),
                elapsed_years=np.asarray(data["elapsed_years"], dtype=np.float32),
                image_paths=np.asarray(data["image_paths"]).astype(str),
                image_hashes=np.asarray(data["image_hashes"]).astype(str),
                mask_hashes=np.asarray(data["mask_hashes"]).astype(str),
                qc_json=np.asarray(data["qc_json"]).astype(str),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
        artifact.validate()
        return artifact
