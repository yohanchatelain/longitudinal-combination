from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable
import warnings

import numpy as np
from scipy import ndimage, signal, special

from .config import ArchitectureConfig, ScatteringConfig
from .image import PreparedImage


@dataclass(frozen=True)
class FeatureVector:
    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or len(self.values) != len(self.names):
            raise ValueError("Feature values and names must be aligned one-dimensional arrays")
        if not np.isfinite(self.values).all():
            raise ValueError("Feature vector contains non-finite values")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature names must be unique")


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _physical_kernel_coordinates(
    sigma_mm: float,
    spacing_mm: tuple[float, float, float],
    truncate_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_mm = truncate_sigma * sigma_mm
    radii = [max(1, int(math.ceil(radius_mm / spacing))) for spacing in spacing_mm]
    axes = [
        np.arange(-radius, radius + 1, dtype=np.float64) * spacing
        for radius, spacing in zip(radii, spacing_mm)
    ]
    return tuple(np.meshgrid(*axes, indexing="ij"))  # type: ignore[return-value]


def _solid_harmonic_kernel(
    sigma_mm: float,
    angular_order: int,
    m: int,
    spacing_mm: tuple[float, float, float],
    truncate_sigma: float,
) -> np.ndarray:
    x, y, z = _physical_kernel_coordinates(sigma_mm, spacing_mm, truncate_sigma)
    radius = np.sqrt(x * x + y * y + z * z)
    safe_radius = np.where(radius > 0, radius, 1.0)
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)
    polar = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if hasattr(special, "sph_harm_y"):
            # SciPy >= 1.15: theta is polar and phi is azimuthal.
            harmonic = special.sph_harm_y(angular_order, m, polar, azimuth)
        else:
            # SciPy < 1.15 used the reversed angular convention.
            harmonic = special.sph_harm(m, angular_order, azimuth, polar)
    radial = np.power(radius / sigma_mm, angular_order) * np.exp(
        -0.5 * np.square(radius / sigma_mm)
    )
    kernel = radial * harmonic
    if angular_order > 0:
        kernel[radius == 0] = 0
    else:
        # Make l=0 a band-pass wavelet rather than a Gaussian low-pass.
        kernel = kernel - kernel.mean()
    voxel_volume = float(np.prod(spacing_mm))
    norm = float(np.sqrt(np.sum(np.abs(kernel) ** 2) * voxel_volume))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Degenerate solid-harmonic kernel for sigma={sigma_mm}, l={angular_order}, m={m}"
        )
    return (kernel / norm).astype(np.complex64)


def _convolve(image: np.ndarray, kernel: np.ndarray, voxel_volume: float) -> np.ndarray:
    response = signal.fftconvolve(image, kernel, mode="same")
    return np.asarray(response * voxel_volume, dtype=np.complex64)


def _orientation_energy_scipy(
    image: np.ndarray,
    sigma_mm: float,
    angular_order: int,
    spacing_mm: tuple[float, float, float],
    truncate_sigma: float,
) -> np.ndarray:
    energy = np.zeros(image.shape, dtype=np.float32)
    voxel_volume = float(np.prod(spacing_mm))
    for m in range(-angular_order, angular_order + 1):
        kernel = _solid_harmonic_kernel(
            sigma_mm,
            angular_order,
            m,
            spacing_mm,
            truncate_sigma,
        )
        response = _convolve(image, kernel, voxel_volume)
        energy += np.square(np.abs(response)).astype(np.float32)
    return np.sqrt(np.maximum(energy, 0), dtype=np.float32)


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The torch scattering backend was requested but PyTorch is unavailable"
        ) from exc
    return torch


def _resolve_runtime(config: ScatteringConfig) -> tuple[str, str]:
    if config.backend == "scipy":
        return "scipy", "cpu"
    try:
        torch = _load_torch()
    except RuntimeError:
        if config.backend == "torch":
            raise
        return "scipy", "cpu"
    if config.device == "auto":
        if torch.cuda.is_available():
            return "torch", "cuda"
        return ("torch", "cpu") if config.backend == "torch" else ("scipy", "cpu")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {config.device!r} requested but CUDA is unavailable")
    return "torch", config.device


def _torch_weight(
    sigma_mm: float,
    angular_order: int,
    spacing_mm: tuple[float, float, float],
    truncate_sigma: float,
    *,
    device: str,
    cache: dict[tuple[Any, ...], Any],
):
    torch = _load_torch()
    key = (sigma_mm, angular_order, spacing_mm, truncate_sigma, device)
    if key in cache:
        return cache[key]
    channels: list[np.ndarray] = []
    for m in range(-angular_order, angular_order + 1):
        kernel = _solid_harmonic_kernel(
            sigma_mm,
            angular_order,
            m,
            spacing_mm,
            truncate_sigma,
        )
        channels.extend(
            [
                np.ascontiguousarray(kernel.real, dtype=np.float32),
                np.ascontiguousarray(kernel.imag, dtype=np.float32),
            ]
        )
    weight = torch.as_tensor(np.stack(channels)[:, None], dtype=torch.float32, device=device)
    cache[key] = weight
    return weight


def _orientation_energy_torch(
    image,
    sigma_mm: float,
    angular_order: int,
    spacing_mm: tuple[float, float, float],
    truncate_sigma: float,
    *,
    device: str,
    cache: dict[tuple[Any, ...], Any],
    convolution: str,
):
    torch = _load_torch()
    import torch.nn.functional as functional

    weight = _torch_weight(
        sigma_mm,
        angular_order,
        spacing_mm,
        truncate_sigma,
        device=device,
        cache=cache,
    )
    kernel_shape = tuple(int(value) for value in weight.shape[-3:])
    if convolution == "auto":
        # Direct cuDNN convolution is excellent for compact stencils; physical
        # wavelets quickly become large enough that linear FFT convolution wins.
        convolution = "fft" if int(np.prod(kernel_shape)) > 9 ** 3 else "direct"
    if convolution == "direct":
        padding = tuple(size // 2 for size in kernel_shape)
        # conv3d is cross-correlation, hence the spatial flip.
        direct_weight = torch.flip(weight, dims=(-3, -2, -1))
        response = functional.conv3d(image[None, None], direct_weight, padding=padding)[0]
    else:
        image_shape = tuple(int(value) for value in image.shape)
        full_shape = tuple(n + k - 1 for n, k in zip(image_shape, kernel_shape))
        fft_key = ("fft", id(weight), image_shape, device)
        kernel_fft = cache.get(fft_key)
        if kernel_fft is None:
            kernel_fft = torch.fft.rfftn(
                weight[:, 0], s=full_shape, dim=(-3, -2, -1)
            )
            cache[fft_key] = kernel_fft
        image_fft = torch.fft.rfftn(image, s=full_shape, dim=(-3, -2, -1))
        full_response = torch.fft.irfftn(
            kernel_fft * image_fft[None], s=full_shape, dim=(-3, -2, -1)
        )
        starts = tuple((size - 1) // 2 for size in kernel_shape)
        response = full_response[
            :,
            starts[0] : starts[0] + image_shape[0],
            starts[1] : starts[1] + image_shape[1],
            starts[2] : starts[2] + image_shape[2],
        ]
    response = response * float(np.prod(spacing_mm))
    energy = torch.sqrt(torch.clamp(torch.sum(response * response, dim=0), min=0.0))
    return energy


def _analysis_mask(mask: np.ndarray, spacing_mm: tuple[float, float, float], erosion_mm: float) -> np.ndarray:
    if erosion_mm <= 0:
        return mask.astype(bool, copy=True)
    distance = ndimage.distance_transform_edt(mask, sampling=spacing_mm)
    eroded = mask & (distance > erosion_mm)
    if not eroded.any():
        raise ValueError("Boundary erosion removed the entire brain mask")
    return eroded


def _region_masks(
    mask: np.ndarray,
    analysis_mask: np.ndarray,
    affine: np.ndarray,
    n_shells: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    indices = np.argwhere(mask)
    homogeneous = np.column_stack([indices, np.ones(len(indices), dtype=float)])
    world = (affine @ homogeneous.T).T[:, :3]
    centroid = world.mean(axis=0)
    distances = np.linalg.norm(world - centroid, axis=1)
    max_distance = float(distances.max())
    if max_distance <= 0:
        raise ValueError("Cannot define radial shells for a point-like mask")
    shell_index = np.minimum((distances / max_distance * n_shells).astype(int), n_shells - 1)
    shell_volume = np.full(mask.shape, -1, dtype=np.int16)
    shell_volume[tuple(indices.T)] = shell_index
    regions: list[tuple[str, np.ndarray]] = [("whole", analysis_mask)]
    for shell in range(n_shells):
        region = analysis_mask & (shell_volume == shell)
        if not region.any():
            raise ValueError(f"Radial shell {shell} is empty after boundary erosion")
        regions.append((f"shell{shell + 1}of{n_shells}", region))
    return tuple(regions)


def _summaries(
    field: np.ndarray,
    regions: Iterable[tuple[str, np.ndarray]],
    stats: tuple[str, ...],
    prefix: str,
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for region_name, region_mask in regions:
        selected = np.asarray(field[region_mask], dtype=np.float64)
        if selected.size == 0 or not np.isfinite(selected).all():
            raise ValueError(f"Invalid values in feature field {prefix}, region {region_name}")
        for stat in stats:
            if stat == "mean":
                value = float(selected.mean())
            elif stat == "std":
                value = float(selected.std(ddof=0))
            elif stat == "rms":
                value = float(np.sqrt(np.mean(np.square(selected))))
            else:
                raise ValueError(f"Unsupported summary statistic: {stat}")
            values.append(value)
            names.append(f"{prefix}|region={region_name}|stat={stat}")
    return values, names


def _mask_covariates(
    image: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
) -> tuple[list[float], list[str]]:
    indices = np.argwhere(mask)
    homogeneous = np.column_stack([indices, np.ones(len(indices), dtype=float)])
    world = (affine @ homogeneous.T).T[:, :3]
    centered = world - world.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered))
    principal_lengths = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0))[::-1]
    volume = float(len(indices) * abs(np.linalg.det(affine[:3, :3])))
    quantiles = np.percentile(image[mask], [10.0, 50.0, 90.0])
    values = [volume, *(float(value) for value in principal_lengths), *(float(value) for value in quantiles)]
    names = [
        "covariate|mask_volume_mm3",
        "covariate|mask_principal_length1_mm",
        "covariate|mask_principal_length2_mm",
        "covariate|mask_principal_length3_mm",
        "covariate|intensity_q10",
        "covariate|intensity_q50",
        "covariate|intensity_q90",
    ]
    return values, names


class PhysicalScatteringExtractor:
    """Deterministic solid-harmonic energy features on a native physical grid."""

    def __init__(self, config: ArchitectureConfig | ScatteringConfig):
        self.config = config.scattering if isinstance(config, ArchitectureConfig) else config
        self.backend, self.device = _resolve_runtime(self.config)
        self._torch_weight_cache: dict[tuple[Any, ...], Any] = {}
        if self.backend == "torch":
            torch = _load_torch()
            torch.use_deterministic_algorithms(self.config.deterministic)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = self.config.deterministic
                torch.backends.cudnn.benchmark = not self.config.deterministic
                torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False

    @property
    def runtime(self) -> dict[str, Any]:
        runtime: dict[str, Any] = {
            "backend": self.backend,
            "device": self.device,
            "deterministic": self.config.deterministic,
            "torch_convolution": self.config.torch_convolution,
        }
        if self.backend == "torch":
            torch = _load_torch()
            runtime["torch_version"] = torch.__version__
            runtime["cuda_version"] = torch.version.cuda
            if self.device.startswith("cuda"):
                runtime["device_name"] = torch.cuda.get_device_name(torch.device(self.device))
        return runtime

    @staticmethod
    def _as_numpy(field: Any) -> np.ndarray:
        if isinstance(field, np.ndarray):
            return field
        return field.detach().cpu().numpy()

    def _energy(
        self,
        image: Any,
        sigma_mm: float,
        angular_order: int,
        spacing: tuple[float, float, float],
    ) -> Any:
        if self.backend == "torch":
            return _orientation_energy_torch(
                image,
                sigma_mm,
                angular_order,
                spacing,
                self.config.kernel_truncate_sigma,
                device=self.device,
                cache=self._torch_weight_cache,
                convolution=self.config.torch_convolution,
            )
        return _orientation_energy_scipy(
            image,
            sigma_mm,
            angular_order,
            spacing,
            self.config.kernel_truncate_sigma,
        )

    def extract(
        self,
        prepared: PreparedImage | None = None,
        *,
        image: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> FeatureVector:
        if prepared is not None:
            image, mask, affine = prepared.image, prepared.mask, prepared.affine
            spacing = prepared.qc.spacing_mm
        else:
            if image is None or mask is None or affine is None:
                raise ValueError("Provide either PreparedImage or image, mask, and affine")
            basis = np.asarray(affine[:3, :3], dtype=float)
            spacing = tuple(float(value) for value in np.linalg.norm(basis, axis=0))
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        affine = np.asarray(affine, dtype=float)
        if image.shape != mask.shape or image.ndim != 3:
            raise ValueError("Scattering input image and mask must be aligned 3D arrays")
        cfg = self.config
        # Native grids can differ between visits. Bound GPU memory by retaining
        # FFT plans/weights only within one extraction instead of accumulating a
        # cache entry for every observed shape and spacing.
        if self.backend == "torch":
            self._torch_weight_cache.clear()
        analysis_mask = _analysis_mask(mask, spacing, cfg.boundary_erosion_mm)
        regions = _region_masks(mask, analysis_mask, affine, cfg.radial_shells)
        all_values: list[float] = []
        all_names: list[str] = []

        values, names = _summaries(image, regions, cfg.summary_stats, "order=0|field=intensity")
        all_values.extend(values)
        all_names.extend(names)

        if self.backend == "torch":
            torch = _load_torch()
            backend_image: Any = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        else:
            backend_image = image

        first_order: dict[tuple[int, int], Any] = {}
        for scale_index, sigma_mm in enumerate(cfg.scales_mm):
            for angular_order in cfg.angular_orders:
                energy = self._energy(backend_image, sigma_mm, angular_order, spacing)
                first_order[(scale_index, angular_order)] = energy
                prefix = f"order=1|j={scale_index}|sigma_mm={_tag(sigma_mm)}|l={angular_order}"
                values, names = _summaries(
                    self._as_numpy(energy), regions, cfg.summary_stats, prefix
                )
                all_values.extend(values)
                all_names.extend(names)

        if cfg.max_order >= 2:
            for (first_scale, first_l), first_energy in first_order.items():
                for second_scale in range(first_scale + 1, len(cfg.scales_mm)):
                    sigma_mm = cfg.scales_mm[second_scale]
                    for second_l in cfg.angular_orders:
                        energy = self._energy(
                            first_energy,
                            sigma_mm,
                            second_l,
                            spacing,
                        )
                        prefix = (
                            f"order=2|j1={first_scale}|l1={first_l}|j2={second_scale}|"
                            f"sigma2_mm={_tag(sigma_mm)}|l2={second_l}"
                        )
                        values, names = _summaries(
                            self._as_numpy(energy), regions, cfg.summary_stats, prefix
                        )
                        all_values.extend(values)
                        all_names.extend(names)

        values, names = _mask_covariates(image, mask, affine)
        all_values.extend(values)
        all_names.extend(names)
        dtype = np.dtype(cfg.dtype)
        return FeatureVector(np.asarray(all_values, dtype=dtype), tuple(all_names))
