from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json_atomic
from .config import ArchitectureConfig, load_config
from .image import load_and_prepare


def mask_overlap(mask_a_path: str | Path, mask_b_path: str | Path) -> dict[str, float]:
    """Compare two masks that are already defined on the same scanner grid."""
    image_a = nib.as_closest_canonical(nib.squeeze_image(nib.load(str(mask_a_path))))
    image_b = nib.as_closest_canonical(nib.squeeze_image(nib.load(str(mask_b_path))))
    if image_a.shape != image_b.shape or not np.allclose(
        image_a.affine, image_b.affine, atol=1e-4
    ):
        raise ValueError("Masks do not share the same canonical grid and affine")
    mask_a = np.asarray(image_a.dataobj) > 0
    mask_b = np.asarray(image_b.dataobj) > 0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    size_a = int(mask_a.sum())
    size_b = int(mask_b.sum())
    return {
        "dice": float(2 * intersection / max(1, size_a + size_b)),
        "jaccard": float(intersection / max(1, union)),
        "reference_voxels": size_a,
        "synthstrip_voxels": size_b,
        "synthstrip_to_reference_volume_ratio": float(size_b / max(1, size_a)),
    }


def _run_synthstrip(
    *,
    executable: str,
    container: Path,
    image_path: Path,
    mask_path: Path,
    gpu: bool,
    threads: int,
) -> tuple[list[str], float]:
    command = [
        executable,
        "exec",
        "--userns",
    ]
    if gpu:
        command.append("--nv")
    command.extend(
        [
            "--bind",
            "/mnt/lustre:/mnt/lustre",
            str(container),
            "mri_synthstrip",
            "-i",
            str(image_path),
            "-m",
            str(mask_path),
            "-t",
            str(threads),
        ]
    )
    if gpu:
        command.append("-g")
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return command, time.perf_counter() - started


def build_synthstrip_manifest(
    source_manifest: str | Path,
    output_dir: str | Path,
    container: str | Path,
    config: ArchitectureConfig,
    *,
    expected_container_sha256: str,
    executable: str = "apptainer",
    gpu: bool = True,
    threads: int = 4,
    overwrite: bool = False,
) -> Path:
    source_path = Path(source_manifest).resolve()
    destination = Path(output_dir).resolve()
    container_path = Path(container).resolve()
    actual_container_sha256 = sha256_file(container_path)
    if actual_container_sha256 != expected_container_sha256:
        raise ValueError(
            "SynthStrip container checksum mismatch: "
            f"expected {expected_container_sha256}, got {actual_container_sha256}"
        )
    table = pd.read_csv(source_path)
    required = {"subject_id", "visit_id", "elapsed_years", "image_path"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Source manifest is missing columns: {missing}")
    masks_dir = destination / "synthstrip_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = destination / "synthstrip_manifest.provenance.json"
    previous_scans: dict[tuple[str, str], dict] = {}
    if provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        previous_scans = {
            (str(scan["subject_id"]), str(scan["visit_id"])): scan
            for scan in previous.get("scans", [])
        }
    records: list[dict] = []
    provenance: list[dict] = []
    for _, row in table.iterrows():
        subject_id = str(row["subject_id"])
        visit_id = str(row["visit_id"])
        image_path = Path(str(row["image_path"])).resolve()
        has_reference = "mask_path" in table.columns and pd.notna(row.get("mask_path"))
        reference_mask_path = (
            Path(str(row["mask_path"])).resolve() if has_reference else None
        )
        mask_path = masks_dir / f"{subject_id}_{visit_id}_desc-synthstrip18_mask.nii.gz"
        prior = previous_scans.get((subject_id, visit_id), {})
        command: list[str] | None = prior.get("command")
        runtime_seconds = float(prior.get("runtime_seconds", 0.0))
        if overwrite or not mask_path.is_file():
            command, runtime_seconds = _run_synthstrip(
                executable=executable,
                container=container_path,
                image_path=image_path,
                mask_path=mask_path,
                gpu=gpu,
                threads=threads,
            )
        prepared = load_and_prepare(image_path, mask_path, config)
        overlap = (
            mask_overlap(reference_mask_path, mask_path)
            if reference_mask_path is not None
            else None
        )
        selection_eligible = bool(row.get("confirmatory_eligible", False))
        record = row.to_dict()
        record["mask_path"] = str(mask_path.resolve())
        record["mask_method"] = "SynthStrip 1.8 (frozen container)"
        record["mask_confirmatory_eligible"] = True
        record["confirmatory_eligible"] = selection_eligible
        records.append(record)
        scan_provenance = {
            "subject_id": subject_id,
            "visit_id": visit_id,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": sha256_file(mask_path),
            "overlap_with_development_mask": overlap,
            "qc": prepared.qc.to_dict(),
            "runtime_seconds": runtime_seconds,
            "command": command,
            "selection_confirmatory_eligible": selection_eligible,
        }
        if reference_mask_path is not None:
            scan_provenance.update(
                {
                    "reference_development_mask": str(reference_mask_path),
                    "reference_development_mask_sha256": sha256_file(reference_mask_path),
                }
            )
        provenance.append(scan_provenance)
        overlap_text = "no reference"
        if overlap is not None:
            overlap_text = f"Dice={overlap['dice']:.4f}"
        print(
            f"[{len(records)}/{len(table)}] {subject_id}/{visit_id}: "
            f"{runtime_seconds:.2f}s, {overlap_text}, "
            f"volume={prepared.qc.mask_volume_mm3:.0f} mm3",
            flush=True,
        )
    output_manifest = destination / "synthstrip_manifest.csv"
    pd.DataFrame(records).to_csv(output_manifest, index=False)
    write_json_atomic(
        provenance_path,
        {
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "container": str(container_path),
            "container_sha256": actual_container_sha256,
            "container_source": "docker://freesurfer/synthstrip:1.8",
            "container_upstream_manifest_digest": (
                "sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06"
            ),
            "gpu_requested": gpu,
            "threads_for_new_runs": threads,
            "n_visits": len(records),
            "mask_confirmatory_eligible": True,
            "selection_confirmatory_eligible": bool(
                records and all(record["confirmatory_eligible"] for record in records)
            ),
            "scans": provenance,
        },
    )
    return output_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--executable", default="apptainer")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = build_synthstrip_manifest(
        args.source_manifest,
        args.output_dir,
        args.container,
        load_config(args.config),
        expected_container_sha256=args.container_sha256,
        executable=args.executable,
        gpu=not args.cpu,
        threads=args.threads,
        overwrite=args.overwrite,
    )
    print(json.dumps({"manifest": str(output)}, indent=2))


if __name__ == "__main__":
    main()
