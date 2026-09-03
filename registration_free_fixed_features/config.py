from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QCConfig:
    min_spacing_mm: float = 0.5
    max_spacing_mm: float = 3.0
    max_shear_cosine: float = 1e-3
    min_mask_voxels: int = 1_000
    min_mask_volume_mm3: float = 50_000.0
    min_largest_component_fraction: float = 0.98
    max_boundary_fraction: float = 0.02
    min_side_fraction: float = 0.20


@dataclass(frozen=True)
class PreprocessingConfig:
    bias_backend: str = "simpleitk_n4"
    n4_iterations: tuple[int, ...] = (50, 50, 30, 20)
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    normalization: str = "robust_zscore"


@dataclass(frozen=True)
class ScatteringConfig:
    scales_mm: tuple[float, ...] = (2.0, 4.0, 8.0)
    angular_orders: tuple[int, ...] = (0, 1, 2)
    max_order: int = 2
    radial_shells: int = 4
    summary_stats: tuple[str, ...] = ("mean", "std")
    kernel_truncate_sigma: float = 3.5
    boundary_erosion_mm: float = 1.0
    dtype: str = "float32"
    backend: str = "auto"
    device: str = "auto"
    deterministic: bool = True
    torch_convolution: str = "auto"


@dataclass(frozen=True)
class EvaluationConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    random_seed: int = 42
    min_elapsed_years: float = 0.25
    max_elapsed_years: float = 10.0
    elastic_net_alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    elastic_net_l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)
    logistic_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class ArchitectureConfig:
    schema_version: str = "rfff-1"
    confirmatory: bool = False
    qc: QCConfig = field(default_factory=QCConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    scattering: ScatteringConfig = field(default_factory=ScatteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        q = self.qc
        p = self.preprocessing
        s = self.scattering
        e = self.evaluation
        if not (0 < q.min_spacing_mm <= q.max_spacing_mm):
            raise ValueError("QC spacing range must be positive and ordered")
        if not (0 < q.min_side_fraction < 0.5):
            raise ValueError("min_side_fraction must be between 0 and 0.5")
        if p.bias_backend not in {"none", "simpleitk_n4"}:
            raise ValueError(f"Unsupported bias backend: {p.bias_backend}")
        if self.confirmatory and p.bias_backend == "none":
            raise ValueError("Confirmatory extraction requires a pinned N4 backend")
        low, high = p.clip_percentiles
        if not (0 <= low < high <= 100):
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100")
        if p.normalization != "robust_zscore":
            raise ValueError("Only robust_zscore normalization is schema-locked")
        if not s.scales_mm or any(value <= 0 for value in s.scales_mm):
            raise ValueError("scales_mm must contain positive values")
        if tuple(sorted(set(s.scales_mm))) != s.scales_mm:
            raise ValueError("scales_mm must be unique and strictly increasing")
        if not s.angular_orders or any(value < 0 for value in s.angular_orders):
            raise ValueError("angular_orders must be non-negative")
        if tuple(sorted(set(s.angular_orders))) != s.angular_orders:
            raise ValueError("angular_orders must be unique and increasing")
        if s.max_order not in {1, 2}:
            raise ValueError("max_order must be 1 or 2")
        if s.radial_shells < 1:
            raise ValueError("radial_shells must be positive")
        if not s.summary_stats or not set(s.summary_stats) <= {"mean", "std", "rms"}:
            raise ValueError("summary_stats may contain mean, std, and rms")
        if s.backend not in {"auto", "scipy", "torch"}:
            raise ValueError("scattering.backend must be auto, scipy, or torch")
        if s.device != "auto" and s.device != "cpu" and not s.device.startswith("cuda"):
            raise ValueError("scattering.device must be auto, cpu, cuda, or cuda:<index>")
        if s.backend == "scipy" and s.device not in {"auto", "cpu"}:
            raise ValueError("The scipy scattering backend is CPU-only")
        if s.torch_convolution not in {"auto", "direct", "fft"}:
            raise ValueError("torch_convolution must be auto, direct, or fft")
        if e.outer_folds < 2 or e.inner_folds < 2:
            raise ValueError("Nested CV requires at least two inner and outer folds")
        if not (0 < e.min_elapsed_years < e.max_elapsed_years):
            raise ValueError("Elapsed-time bounds must be positive and ordered")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tuple(value: Any) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _construct(raw: dict[str, Any]) -> ArchitectureConfig:
    qc = QCConfig(**raw.get("qc", {}))
    prep_raw = dict(raw.get("preprocessing", {}))
    if "n4_iterations" in prep_raw:
        prep_raw["n4_iterations"] = _tuple(prep_raw["n4_iterations"])
    if "clip_percentiles" in prep_raw:
        prep_raw["clip_percentiles"] = _tuple(prep_raw["clip_percentiles"])
    preprocessing = PreprocessingConfig(**prep_raw)
    scat_raw = dict(raw.get("scattering", {}))
    for key in ("scales_mm", "angular_orders", "summary_stats"):
        if key in scat_raw:
            scat_raw[key] = _tuple(scat_raw[key])
    scattering = ScatteringConfig(**scat_raw)
    eval_raw = dict(raw.get("evaluation", {}))
    for key in ("elastic_net_alphas", "elastic_net_l1_ratios", "logistic_cs"):
        if key in eval_raw:
            eval_raw[key] = _tuple(eval_raw[key])
    evaluation = EvaluationConfig(**eval_raw)
    known = {"schema_version", "confirmatory", "qc", "preprocessing", "scattering", "evaluation"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {unknown}")
    config = ArchitectureConfig(
        schema_version=str(raw.get("schema_version", "rfff-1")),
        confirmatory=bool(raw.get("confirmatory", False)),
        qc=qc,
        preprocessing=preprocessing,
        scattering=scattering,
        evaluation=evaluation,
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ArchitectureConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return _construct(raw)


def config_from_dict(raw: dict[str, Any]) -> ArchitectureConfig:
    return _construct(raw)
