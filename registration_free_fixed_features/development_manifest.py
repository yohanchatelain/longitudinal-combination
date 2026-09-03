from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json_atomic


def _raw_3d(path: Path) -> nib.spatialimages.SpatialImage:
    image = nib.squeeze_image(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected singleton-squeezable raw 3D image at {path}, got {image.shape}")
    return image


def _elapsed_years(rows: pd.DataFrame) -> np.ndarray:
    if "days_since_enrollment" in rows and rows["days_since_enrollment"].notna().all():
        values = rows["days_since_enrollment"].astype(float).to_numpy() / 365.25
    elif "days" in rows and rows["days"].notna().all():
        values = rows["days"].astype(float).to_numpy() / 365.25
    else:
        values = rows["age"].astype(float).to_numpy()
    return values - values.min()


def _geometry_score(image_path: str) -> tuple[float, str]:
    image = _raw_3d(Path(image_path))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    score = float(np.max(np.abs(np.log(spacing))))
    signature = f"{image.shape}:{tuple(np.round(spacing, 4))}"
    return score, signature


def _select_subjects(
    cohort: pd.DataFrame,
    freesurfer_root: Path,
    subjects_per_group: int,
) -> list[str]:
    eligible: list[dict] = []
    for subject, rows in cohort.groupby("participant_id", sort=True):
        if len(rows) < 2:
            continue
        ordered = rows.sort_values(["timepoint", "session"])
        endpoints = ordered.iloc[[0, -1]]
        if not all(Path(path).is_file() for path in endpoints["image_path"]):
            continue
        if not all(
            (freesurfer_root / str(directory) / "mri" / "brainmask.mgz").is_file()
            for directory in endpoints["directory"]
        ):
            continue
        scores = [_geometry_score(path) for path in endpoints["image_path"]]
        eligible.append(
            {
                "subject_id": str(subject),
                "group": str(endpoints.iloc[0]["studies"]),
                "geometry_score": max(score for score, _ in scores),
                "geometry_signature": ";".join(signature for _, signature in scores),
            }
        )
    eligible_frame = pd.DataFrame(eligible)
    if eligible_frame.empty:
        raise ValueError("No subjects have two raw images and two FreeSurfer development masks")
    selected: list[str] = []
    for group, rows in eligible_frame.groupby("group", sort=True):
        ranked = rows.sort_values(
            ["geometry_score", "geometry_signature", "subject_id"],
            ascending=[False, True, True],
        )
        if len(ranked) < subjects_per_group:
            raise ValueError(
                f"Group {group} has only {len(ranked)} eligible subjects; "
                f"requested {subjects_per_group}"
            )
        selected.extend(ranked.head(subjects_per_group)["subject_id"].tolist())
    return selected


def build_development_manifest(
    cohort_csv: str | Path,
    freesurfer_root: str | Path,
    output_dir: str | Path,
    *,
    subjects_per_group: int = 2,
) -> Path:
    cohort_path = Path(cohort_csv).resolve()
    fs_root = Path(freesurfer_root).resolve()
    destination = Path(output_dir).resolve()
    masks_dir = destination / "native_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path)
    required = {"participant_id", "session", "timepoint", "age", "studies", "directory", "image_path"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Cohort table is missing columns: {missing}")
    selected = _select_subjects(cohort, fs_root, subjects_per_group)
    records: list[dict] = []
    mask_provenance: list[dict] = []
    for subject in sorted(selected):
        rows = cohort[cohort["participant_id"].astype(str) == subject].sort_values(
            ["timepoint", "session"]
        )
        rows = rows.iloc[[0, -1]].copy()
        elapsed = _elapsed_years(rows)
        for (_, row), elapsed_years in zip(rows.iterrows(), elapsed):
            raw_path = Path(str(row["image_path"])).resolve()
            fs_mask_path = fs_root / str(row["directory"]) / "mri" / "brainmask.mgz"
            raw_image = _raw_3d(raw_path)
            fs_mask = nib.load(str(fs_mask_path))
            native_mask_image = resample_from_to(fs_mask, raw_image, order=0)
            native_mask = np.asarray(native_mask_image.dataobj) > 0
            if not native_mask.any():
                raise ValueError(f"Empty resampled mask for {subject}/{row['session']}")
            mask_path = masks_dir / f"{subject}_{row['session']}_desc-fsdev_mask.nii.gz"
            header = raw_image.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(
                nib.Nifti1Image(native_mask.astype(np.uint8), raw_image.affine, header),
                mask_path,
            )
            spacing = tuple(float(value) for value in raw_image.header.get_zooms()[:3])
            records.append(
                {
                    "subject_id": subject,
                    "visit_id": str(row["session"]),
                    "elapsed_years": float(elapsed_years),
                    "image_path": str(raw_path),
                    "mask_path": str(mask_path.resolve()),
                    "group": str(row["studies"]),
                    "age": float(row["age"]),
                    "source_directory": str(row["directory"]),
                    "raw_shape": "x".join(str(value) for value in raw_image.shape),
                    "spacing_mm": "x".join(f"{value:.4g}" for value in spacing),
                    "confirmatory_eligible": False,
                }
            )
            mask_provenance.append(
                {
                    "subject_id": subject,
                    "visit_id": str(row["session"]),
                    "raw_image_sha256": sha256_file(raw_path),
                    "freesurfer_mask": str(fs_mask_path.resolve()),
                    "freesurfer_mask_sha256": sha256_file(fs_mask_path),
                    "native_mask": str(mask_path.resolve()),
                    "native_mask_sha256": sha256_file(mask_path),
                    "resampling": "nibabel.processing.resample_from_to order=0 using header geometry",
                    "confirmatory_eligible": False,
                }
            )
    manifest = pd.DataFrame(records).sort_values(["group", "subject_id", "elapsed_years"])
    manifest_path = destination / "development_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_json_atomic(
        destination / "development_manifest.provenance.json",
        {
            "cohort_csv": str(cohort_path),
            "cohort_sha256": sha256_file(cohort_path),
            "freesurfer_root": str(fs_root),
            "subjects_per_group": subjects_per_group,
            "selected_subjects": sorted(selected),
            "n_visits": len(manifest),
            "confirmatory_eligible": False,
            "reason": "Masks derive from FreeSurfer and are permitted only for development/resource pilots",
            "masks": mask_provenance,
        },
    )
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--freesurfer-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects-per-group", type=int, default=2)
    args = parser.parse_args(argv)
    path = build_development_manifest(
        args.cohort_csv,
        args.freesurfer_root,
        args.output_dir,
        subjects_per_group=args.subjects_per_group,
    )
    print(json.dumps({"manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
