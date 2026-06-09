#!/usr/bin/env python3
"""Build PPMI cohort CSV in NKI-compatible format.

Reads:
  - living-park subject-level ROI CSVs  → subject / session list
  - Age_at_visit.csv                    → age per visit
  - Participant_Status.csv              → PD (COHORT=1) / HC (COHORT=2) label
  - Demographics.csv                    → sex
  - living-park/inputs/sub-*/ses-*/anat/PPMI_*.nii → T1w image paths

Outputs:
  PPMI_data/cohort_longitudinal.csv   (NKI cohort schema)

Column mapping:
  id                  = PATNO (numeric string)
  participant_id      = sub-{PATNO}
  session             = EVENT_ID
  session_type        = 'SCAN'
  age                 = AGE_AT_VISIT (years)
  sex                 = SEX from Demographics (0=F, 1=M → 'F'/'M')
  timepoint           = sequential integer per subject ordered by age
  directory           = sub-{PATNO}_ses-{EVENT_ID}
  days                = days from first scan (approx from age delta)
  days_since_enrollment = same
  studies             = {PPMI_PD} or {PPMI_HC}
  study_num           = 1
  image_path          = path to T1w .nii file (relative to living-park root)
  mask_path           = '' (no brain mask available)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd

LIVING_PARK = Path("/mnt/lustre/ychatel/living-park")
STUDY_FILES = LIVING_PARK / "inputs" / "study_files"
INPUTS = LIVING_PARK / "inputs"
ROI_CSV = LIVING_PARK / "csv" / "subject-thickness-mean.csv"

COHORT_PD = 1
COHORT_HC = 2


def find_t1w(sub: str, ses: str) -> str:
    """Return absolute path to raw T1w .nii for a subject/session."""
    anat_dir = INPUTS / sub / ses / "anat"
    raw_niis = sorted(glob.glob(str(anat_dir / "PPMI_*.nii")))
    if not raw_niis:
        return ""
    return raw_niis[0]


def build_cohort(dry_run: bool = False) -> pd.DataFrame:
    # --- Subject/session list from ROI CSV ---
    df_roi = pd.read_csv(ROI_CSV)
    pairs = df_roi["subjid"].unique()
    parsed = []
    for s in pairs:
        m = re.match(r"(sub-\d+)_(ses-\w+)", str(s))
        if m:
            sub, ses = m.group(1), m.group(2)
            patno = int(re.sub(r"sub-", "", sub))
            event_id = re.sub(r"ses-", "", ses)
            parsed.append({"patno": patno, "sub": sub, "ses": ses, "event_id": event_id})
    df = pd.DataFrame(parsed)
    print(f"Subject-session pairs from ROI CSV: {len(df)}")

    # --- Age ---
    age_df = pd.read_csv(STUDY_FILES / "Age_at_visit.csv")
    df = df.merge(
        age_df[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]],
        left_on=["patno", "event_id"],
        right_on=["PATNO", "EVENT_ID"],
        how="left",
    )
    n_missing_age = df["AGE_AT_VISIT"].isna().sum()
    if n_missing_age:
        print(f"  Dropping {n_missing_age} rows with missing age")
        df = df[df["AGE_AT_VISIT"].notna()].copy()

    # --- PD/HC label ---
    status_df = pd.read_csv(STUDY_FILES / "Participant_Status.csv")
    # Keep only PD (1) and HC (2)
    status_pd_hc = status_df[status_df["COHORT"].isin([COHORT_PD, COHORT_HC])][
        ["PATNO", "COHORT"]
    ].drop_duplicates("PATNO")
    df = df.merge(status_pd_hc, on="PATNO", how="inner")
    print(f"After PD/HC filter: {len(df)} rows ({df['patno'].nunique()} subjects)")

    # --- Sex ---
    demo_df = pd.read_csv(STUDY_FILES / "Demographics.csv")
    # Take first row per PATNO (demographics don't change)
    sex_df = demo_df[["PATNO", "SEX"]].drop_duplicates("PATNO")
    df = df.merge(sex_df, on="PATNO", how="left")
    df["sex_str"] = df["SEX"].map({0: "F", 1: "M"}).fillna("U")

    # --- T1w image path ---
    print("Finding T1w images...")
    df["image_path"] = df.apply(lambda r: find_t1w(r["sub"], r["ses"]), axis=1)
    n_no_t1w = (df["image_path"] == "").sum()
    if n_no_t1w:
        print(f"  Dropping {n_no_t1w} rows without T1w")
        df = df[df["image_path"] != ""].copy()
    print(f"  {len(df)} rows with T1w images")

    # --- Timepoint (sequential per subject, ordered by age) ---
    df = df.sort_values(["patno", "AGE_AT_VISIT"]).reset_index(drop=True)
    df["timepoint"] = df.groupby("patno").cumcount() + 1

    # --- Days from first scan ---
    df["age_first"] = df.groupby("patno")["AGE_AT_VISIT"].transform("min")
    df["days_since_enrollment"] = ((df["AGE_AT_VISIT"] - df["age_first"]) * 365.25).round(0).astype(int)
    df["days"] = df["days_since_enrollment"]

    # --- Studies field ---
    df["studies"] = df["COHORT"].map({COHORT_PD: "{PPMI_PD}", COHORT_HC: "{PPMI_HC}"})

    # --- Construct output dataframe ---
    out = pd.DataFrame({
        "id": df["patno"].astype(str),
        "participant_id": df["sub"],
        "session": df["event_id"],
        "session_type": "SCAN",
        "age": df["AGE_AT_VISIT"],
        "sex": df["sex_str"],
        "timepoint": df["timepoint"],
        "directory": df["sub"] + "_" + df["ses"],
        "days": df["days"],
        "days_since_enrollment": df["days_since_enrollment"],
        "studies": df["studies"],
        "study_num": 1,
        "image_path": df["image_path"],
        "mask_path": "",
    })

    # --- Summary ---
    n_pd = (df["COHORT"] == COHORT_PD).sum()
    n_hc = (df["COHORT"] == COHORT_HC).sum()
    n_multi = (df.groupby("patno").size() >= 2).sum()
    print(f"\nCohort summary:")
    print(f"  Total rows (subject-sessions): {len(out)}")
    print(f"  Unique subjects: {out['id'].nunique()}")
    print(f"  PD sessions: {n_pd}, HC sessions: {n_hc}")
    print(f"  Subjects with >=2 timepoints: {n_multi}")
    print(f"  Age range: {df['AGE_AT_VISIT'].min():.1f} – {df['AGE_AT_VISIT'].max():.1f} yr")

    if dry_run:
        out = out[out["id"].isin(out["id"].unique()[:20])].copy()
        print(f"\n[DRY RUN] Keeping first 20 subjects → {len(out)} rows")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PPMI cohort CSV.")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parents[1] / "PPMI_data" / "cohort_longitudinal.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cohort = build_cohort(dry_run=args.dry_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
