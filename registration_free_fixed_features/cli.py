from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .artifacts import (
    VisitFeatureArtifact,
    environment_snapshot,
    git_state,
    sha256_file,
    write_json_atomic,
)
from .config import load_config
from .evaluation import make_outer_folds, run_paired_nested_cv
from .image import load_and_prepare
from .longitudinal import build_candidate_matrices
from .scattering import PhysicalScatteringExtractor


REQUIRED_MANIFEST_COLUMNS = {
    "subject_id",
    "visit_id",
    "elapsed_years",
    "image_path",
    "mask_path",
}


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def extract_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest_path = Path(args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest[["subject_id", "visit_id"]].duplicated().any():
        raise ValueError("Manifest subject_id/visit_id pairs must be unique")
    extractor = PhysicalScatteringExtractor(config)
    rows: list[np.ndarray] = []
    schema: tuple[str, ...] | None = None
    qc_json: list[str] = []
    image_hashes: list[str] = []
    mask_hashes: list[str] = []
    image_paths: list[str] = []
    for visit_index, row in enumerate(manifest.itertuples(index=False), start=1):
        image_path = Path(str(row.image_path)).expanduser().resolve()
        mask_path = Path(str(row.mask_path)).expanduser().resolve()
        print(
            f"[{visit_index}/{len(manifest)}] {row.subject_id}/{row.visit_id}: "
            f"preparing {image_path.name}",
            flush=True,
        )
        started = time.perf_counter()
        prepared = load_and_prepare(image_path, mask_path, config)
        preparation_seconds = time.perf_counter() - started
        peak_cuda_memory = None
        torch = None
        if extractor.backend == "torch" and extractor.device.startswith("cuda"):
            import torch as torch_module

            torch = torch_module
            device = torch.device(extractor.device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        features = extractor.extract(prepared)
        if torch is not None:
            torch.cuda.synchronize(torch.device(extractor.device))
            peak_cuda_memory = int(torch.cuda.max_memory_allocated(torch.device(extractor.device)))
        extraction_seconds = time.perf_counter() - started
        if schema is None:
            schema = features.names
        elif schema != features.names:
            raise RuntimeError(f"Feature schema drift at {image_path}")
        rows.append(features.values)
        qc_record = prepared.qc.to_dict()
        qc_record.update(
            {
                "preparation_seconds": preparation_seconds,
                "extraction_seconds": extraction_seconds,
                "peak_cuda_memory_bytes": peak_cuda_memory,
                "corrected_image_sha256": prepared.corrected_image_sha256,
                "corrected_image_hash_contract": (
                    "sha256(shape-ascii + '|float32-le|' + canonical-RAS C-order bytes)"
                ),
            }
        )
        qc_json.append(json.dumps(qc_record, sort_keys=True))
        image_hashes.append(sha256_file(image_path))
        mask_hashes.append(sha256_file(mask_path))
        image_paths.append(str(image_path))
        print(
            f"[{visit_index}/{len(manifest)}] complete: prep={preparation_seconds:.2f}s, "
            f"features={extraction_seconds:.2f}s, n_features={len(features.values)}, "
            f"peak_cuda_mib={0 if peak_cuda_memory is None else peak_cuda_memory / 2**20:.1f}",
            flush=True,
        )
    if not rows or schema is None:
        raise ValueError("Manifest contains no visits")
    repository = Path(__file__).resolve().parents[1]
    artifact = VisitFeatureArtifact(
        X=np.vstack(rows).astype(np.float32),
        feature_names=schema,
        subject_ids=manifest["subject_id"].astype(str).to_numpy(),
        visit_ids=manifest["visit_id"].astype(str).to_numpy(),
        elapsed_years=manifest["elapsed_years"].astype(float).to_numpy(),
        image_paths=np.asarray(image_paths),
        image_hashes=np.asarray(image_hashes),
        mask_hashes=np.asarray(mask_hashes),
        qc_json=np.asarray(qc_json),
        metadata={
            "schema_version": config.schema_version,
            "config_hash": config.hash,
            "resolved_config": config.to_dict(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "environment": environment_snapshot(),
            "git": git_state(repository),
            "scattering_runtime": extractor.runtime,
        },
    )
    artifact.save(args.output)


def folds_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    targets = pd.read_csv(args.targets)
    if not {"subject_id", args.target_column} <= set(targets.columns):
        raise ValueError("Target table must contain subject_id and the requested target column")
    if targets["subject_id"].astype(str).duplicated().any():
        raise ValueError("Target table must contain one row per subject")
    subjects = targets["subject_id"].astype(str).to_numpy()
    if args.task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(targets[args.target_column].astype(str))
        label_mapping = {str(label): int(index) for index, label in enumerate(encoder.classes_)}
    else:
        y = targets[args.target_column].astype(float).to_numpy()
        label_mapping = None
    folds = make_outer_folds(
        subjects,
        y,
        task=args.task,
        n_splits=config.evaluation.outer_folds,
        random_seed=config.evaluation.random_seed,
    )
    output = pd.DataFrame({"subject_id": subjects, "fold": folds})
    _write_csv_atomic(output, Path(args.output))
    write_json_atomic(
        str(args.output) + ".json",
        {
            "config_hash": config.hash,
            "task": args.task,
            "target_column": args.target_column,
            "label_mapping": label_mapping,
        },
    )


def evaluate_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    artifact = VisitFeatureArtifact.load(args.features)
    subjects, candidates, _ = build_candidate_matrices(
        artifact.X,
        artifact.subject_ids,
        artifact.elapsed_years,
        artifact.feature_names,
        min_elapsed_years=config.evaluation.min_elapsed_years,
        max_elapsed_years=config.evaluation.max_elapsed_years,
    )
    targets = pd.read_csv(args.targets)
    target_series = targets.assign(subject_id=targets["subject_id"].astype(str)).set_index("subject_id")[
        args.target_column
    ]
    missing = sorted(set(subjects) - set(target_series.index))
    if missing:
        raise ValueError(f"Missing targets for {len(missing)} subject(s): {missing[:5]}")
    raw_y = target_series.loc[subjects]
    label_mapping = None
    if args.task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_y.astype(str))
        label_mapping = {str(label): int(index) for index, label in enumerate(encoder.classes_)}
    else:
        y = raw_y.astype(float).to_numpy()
    fold_ids = None
    if args.folds:
        saved = pd.read_csv(args.folds).assign(subject_id=lambda frame: frame.subject_id.astype(str))
        fold_series = saved.set_index("subject_id")["fold"]
        if not set(subjects) <= set(fold_series.index):
            raise ValueError("Saved fold file does not cover every evaluated subject")
        fold_ids = fold_series.loc[subjects].to_numpy(dtype=int)
    evaluation = run_paired_nested_cv(
        candidates,
        y,
        subjects,
        task=args.task,
        config=config.evaluation,
        fold_ids=fold_ids,
    )
    output_dir = Path(args.output_dir)
    _write_csv_atomic(evaluation.fold_metrics, output_dir / "fold_metrics.csv")
    _write_csv_atomic(evaluation.predictions, output_dir / "predictions.csv")
    _write_csv_atomic(evaluation.fold_assignments, output_dir / "fold_assignments.csv")
    write_json_atomic(
        output_dir / "evaluation.json",
        {
            "config_hash": config.hash,
            "feature_artifact_sha256": sha256_file(args.features),
            "target_table_sha256": sha256_file(args.targets),
            "task": args.task,
            "target_column": args.target_column,
            "candidates": sorted(candidates),
            "label_mapping": label_mapping,
        },
    )


def gpu_smoke_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    extractor = PhysicalScatteringExtractor(config)
    if extractor.backend != "torch" or not extractor.device.startswith("cuda"):
        raise RuntimeError(
            f"GPU smoke test resolved {extractor.backend}/{extractor.device}; "
            "run on a CUDA node or set scattering.backend=torch and device=cuda"
        )
    import torch

    size = int(args.size)
    if size < 17:
        raise ValueError("GPU smoke phantom size must be at least 17")
    axes = np.arange(size, dtype=np.float32) - (size - 1) / 2
    x, y_axis, z = np.meshgrid(axes, axes, axes, indexing="ij")
    radius = np.sqrt(x * x + y_axis * y_axis + z * z)
    mask = radius <= size * 0.32
    image = np.zeros((size, size, size), dtype=np.float32)
    image[mask] = (
        np.exp(-np.square(radius[mask]) / (0.08 * size * size))
        + 0.2 * (x[mask] > 0)
        + 0.1 * (y_axis[mask] > 0)
    )
    affine = np.eye(4)
    device = torch.device(extractor.device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    gpu_features = extractor.extract(image=image, mask=mask, affine=affine)
    torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - started
    report: dict[str, object] = {
        "runtime": extractor.runtime,
        "shape": list(image.shape),
        "n_features": len(gpu_features.values),
        "gpu_seconds": gpu_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "finite": bool(np.isfinite(gpu_features.values).all()),
    }
    if args.compare_cpu:
        cpu_config = replace(config.scattering, backend="scipy", device="cpu")
        cpu_extractor = PhysicalScatteringExtractor(cpu_config)
        started = time.perf_counter()
        cpu_features = cpu_extractor.extract(image=image, mask=mask, affine=affine)
        cpu_seconds = time.perf_counter() - started
        if gpu_features.names != cpu_features.names:
            raise RuntimeError("CUDA and SciPy feature schemas differ")
        absolute = np.abs(gpu_features.values.astype(float) - cpu_features.values.astype(float))
        denominator = max(float(np.linalg.norm(cpu_features.values)), 1e-12)
        report.update(
            {
                "cpu_seconds": cpu_seconds,
                "speedup": cpu_seconds / max(gpu_seconds, 1e-12),
                "relative_l2_error": float(np.linalg.norm(absolute) / denominator),
                "max_absolute_error": float(absolute.max(initial=0.0)),
            }
        )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="Extract immutable visit-level features")
    extract.add_argument("--config", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=extract_command)

    folds = subparsers.add_parser("make-folds", help="Create reusable subject-level outer folds")
    folds.add_argument("--config", required=True)
    folds.add_argument("--targets", required=True)
    folds.add_argument("--target-column", required=True)
    folds.add_argument("--task", choices=("regression", "classification"), required=True)
    folds.add_argument("--output", required=True)
    folds.set_defaults(func=folds_command)

    evaluate = subparsers.add_parser("evaluate", help="Run paired nested cross-validation")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--target-column", required=True)
    evaluate.add_argument("--task", choices=("regression", "classification"), required=True)
    evaluate.add_argument("--folds")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=evaluate_command)

    gpu_smoke = subparsers.add_parser(
        "gpu-smoke", help="Benchmark CUDA extraction on an analytic phantom"
    )
    gpu_smoke.add_argument("--config", required=True)
    gpu_smoke.add_argument("--size", type=int, default=48)
    gpu_smoke.add_argument("--compare-cpu", action="store_true")
    gpu_smoke.add_argument("--output")
    gpu_smoke.set_defaults(func=gpu_smoke_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
