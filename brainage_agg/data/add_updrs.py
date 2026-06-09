"""Merge clinical scores into the PPMI manifest.

Adds the following columns to manifest_with_updrs.csv for each subject:

  Motor (MDS-UPDRS Part III):
    NP3TOT_first       — motor score at first imaging session (OFF-state preferred)
    NP3TOT_last        — motor score at last imaging session
    NP3TOT_change      — NP3TOT_last - NP3TOT_first  (progression)
    NP3TOT_change_rate — NP3TOT_change / delta_t  (annualized progression, per year)
    PDSTATE_first      — medication state at first session ('OFF', 'ON', or NaN)
    PDSTATE_last       — medication state at last session
    NHY_first          — Hoehn & Yahr stage at first session (OFF-state preferred; 101=NaN)
    NHY_last           — Hoehn & Yahr stage at last session
    NHY_change         — NHY_last - NHY_first

  Session metadata:
    session_first      — PPMI session code at first imaging timepoint (e.g. 'BL', 'V04')
    session_last       — PPMI session code at last imaging timepoint

  Disease history:
    disease_duration_first — years from PD diagnosis (PDDXDT) to first imaging session

  Cognition (MoCA):
    MOCA_first         — MoCA total score at first session
    MOCA_last          — MoCA total score at last session
    MOCA_change        — MOCA_last - MOCA_first

  Demographics:
    sex                — biological sex from cohort file

Usage:
    python3 -m brainage_agg.data.add_updrs
    python3 brainage_agg/data/add_updrs.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LIVING_PARK  = Path("/mnt/lustre/ychatel/living-park")
STUDY_FILES  = LIVING_PARK / "inputs" / "study_files"
UPDRS_PATH   = STUDY_FILES / "MDS_UPDRS_Part_III_clean.csv"
MOCA_PATH    = STUDY_FILES / "Montreal_Cognitive_Assessment__MoCA_.csv"
DIAG_PATH    = STUDY_FILES / "PD_Diagnosis_History.csv"
DEMO_PATH    = STUDY_FILES / "Demographics.csv"
AGE_PATH     = STUDY_FILES / "Age_at_visit.csv"

PROJECT_ROOT = Path(__file__).parents[2]
COHORT_PATH  = PROJECT_ROOT / "PPMI_data" / "cohort_longitudinal.csv"
MANIFEST_IN  = PROJECT_ROOT / "brainage_agg" / "ppmi_outputs" / "manifest.csv"
MANIFEST_OUT = PROJECT_ROOT / "brainage_agg" / "ppmi_outputs" / "manifest_with_updrs.csv"

# NHY=101 is the PPMI sentinel code for "unable to assess in OFF state"
_NHY_INVALID = 101.0


def _parse_month_year(s: str) -> float:
    """Parse 'MM/YYYY' → decimal year (mid-month)."""
    month, year = str(s).split("/")
    return int(year) + (int(month) - 0.5) / 12.0


def _best_updrs_session(rows: pd.DataFrame) -> dict:
    """Return NP3TOT, NHY, PDSTATE from one subject+session.

    Prefers OFF-medication state; among ties, picks the row with valid NP3TOT.
    NHY=101 (unable to assess) is returned as NaN.
    """
    out = {"NP3TOT": np.nan, "NHY": np.nan, "PDSTATE": np.nan}
    if rows.empty:
        return out
    off = rows[rows["PDSTATE"] == "OFF"]
    pool = off if not off.empty else rows
    pool = pool.dropna(subset=["NP3TOT"])
    if pool.empty:
        return out
    row = pool.iloc[0]
    out["NP3TOT"]  = float(row["NP3TOT"])
    out["PDSTATE"] = str(row["PDSTATE"])
    nhy = row.get("NHY", np.nan)
    if pd.notna(nhy) and float(nhy) != _NHY_INVALID:
        out["NHY"] = float(nhy)
    return out


def _best_moca_session(rows: pd.DataFrame) -> float:
    """Return MCATOT from one subject+session (first valid row)."""
    pool = rows.dropna(subset=["MCATOT"])
    if pool.empty:
        return np.nan
    return float(pool.iloc[0]["MCATOT"])


def build_manifest_with_updrs(
    cohort_path:  Path = COHORT_PATH,
    updrs_path:   Path = UPDRS_PATH,
    moca_path:    Path = MOCA_PATH,
    diag_path:    Path = DIAG_PATH,
    demo_path:    Path = DEMO_PATH,
    age_path:     Path = AGE_PATH,
    manifest_in:  Path = MANIFEST_IN,
    manifest_out: Path = MANIFEST_OUT,
) -> pd.DataFrame:
    cohort   = pd.read_csv(cohort_path)
    updrs    = pd.read_csv(updrs_path)
    moca     = pd.read_csv(moca_path)
    diag     = pd.read_csv(diag_path)
    demo     = pd.read_csv(demo_path)
    age_vis  = pd.read_csv(age_path)
    manifest = pd.read_csv(manifest_in)

    cohort["patno"] = cohort["participant_id"].str.replace("sub-", "").astype(int)

    # subject → {timepoint: session_code}
    ses_map = (
        cohort.groupby(["patno", "timepoint"])["session"]
        .first()
        .unstack(fill_value=None)
    )

    # subject → sex
    sex_map = cohort.groupby("patno")["sex"].first()

    # subject → age at each session {(patno, event_id): age}
    age_map: dict[tuple, float] = {
        (int(r.PATNO), r.EVENT_ID): float(r.AGE_AT_VISIT)
        for r in age_vis.itertuples(index=False)
        if pd.notna(r.AGE_AT_VISIT)
    }

    # subject → BIRTHDT (MM/YYYY)
    birth_map: dict[int, str] = (
        demo.dropna(subset=["BIRTHDT"])
        .groupby("PATNO")["BIRTHDT"]
        .first()
        .to_dict()
    )

    # subject → PDDXDT (MM/YYYY)  — use SC visit row (EVENT_ID='SC')
    pddx_map: dict[int, str] = (
        diag.dropna(subset=["PDDXDT"])
        .groupby("PATNO")["PDDXDT"]
        .first()
        .to_dict()
    )

    results = []
    for _, mrow in manifest.iterrows():
        sid   = int(mrow["subject_id"])
        entry: dict = {"subject_id": sid}
        delta_t = float(mrow.get("delta_t", np.nan))

        entry["sex"] = str(sex_map.get(sid, np.nan))

        for label, tp in [("first", 1), ("last", 2)]:
            ses_code = (
                ses_map.loc[sid, tp]
                if (sid in ses_map.index and tp in ses_map.columns)
                else None
            )
            entry[f"session_{label}"] = ses_code

            # --- UPDRS / H&Y ---
            if ses_code is None:
                entry[f"NP3TOT_{label}"]  = np.nan
                entry[f"NHY_{label}"]     = np.nan
                entry[f"PDSTATE_{label}"] = np.nan
            else:
                rows = updrs[(updrs["PATNO"] == sid) & (updrs["EVENT_ID"] == ses_code)]
                scores = _best_updrs_session(rows)
                entry[f"NP3TOT_{label}"]  = scores["NP3TOT"]
                entry[f"NHY_{label}"]     = scores["NHY"]
                entry[f"PDSTATE_{label}"] = scores["PDSTATE"]

            # --- MoCA ---
            if ses_code is None:
                entry[f"MOCA_{label}"] = np.nan
            else:
                mrows = moca[(moca["PATNO"] == sid) & (moca["EVENT_ID"] == ses_code)]
                entry[f"MOCA_{label}"] = _best_moca_session(mrows)

        # --- Derived: NP3TOT change ---
        np3_first = entry["NP3TOT_first"]
        np3_last  = entry["NP3TOT_last"]
        if pd.notna(np3_first) and pd.notna(np3_last):
            change = np3_last - np3_first
            entry["NP3TOT_change"] = change
            entry["NP3TOT_change_rate"] = (
                change / delta_t if (pd.notna(delta_t) and abs(delta_t) > 1e-8) else np.nan
            )
        else:
            entry["NP3TOT_change"]      = np.nan
            entry["NP3TOT_change_rate"] = np.nan

        # --- Derived: NHY change ---
        nhy_first = entry["NHY_first"]
        nhy_last  = entry["NHY_last"]
        entry["NHY_change"] = (
            nhy_last - nhy_first
            if (pd.notna(nhy_first) and pd.notna(nhy_last))
            else np.nan
        )

        # --- Derived: MoCA change ---
        moca_first = entry["MOCA_first"]
        moca_last  = entry["MOCA_last"]
        entry["MOCA_change"] = (
            moca_last - moca_first
            if (pd.notna(moca_first) and pd.notna(moca_last))
            else np.nan
        )

        # --- Disease duration at first session ---
        ses_first = entry["session_first"]
        birth_str = birth_map.get(sid)
        pddx_str  = pddx_map.get(sid)
        age_first = age_map.get((sid, ses_first)) if ses_first else None
        if age_first is not None and birth_str and pddx_str:
            try:
                birth_decimal    = _parse_month_year(birth_str)
                pddx_decimal     = _parse_month_year(pddx_str)
                age_at_diagnosis = pddx_decimal - birth_decimal
                entry["disease_duration_first"] = float(age_first) - age_at_diagnosis
            except (ValueError, TypeError):
                entry["disease_duration_first"] = np.nan
        else:
            entry["disease_duration_first"] = np.nan

        results.append(entry)

    updrs_cols = pd.DataFrame(results).set_index("subject_id")

    manifest_out_df = manifest.copy()
    manifest_out_df = manifest_out_df.join(updrs_cols, on="subject_id")

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out_df.to_csv(manifest_out, index=False)
    print(f"Written: {manifest_out}  ({len(manifest_out_df)} rows)")

    _print_coverage(manifest_out_df)
    return manifest_out_df


def _print_coverage(df: pd.DataFrame) -> None:
    pd_mask = df["band"] == "PD"
    n_pd    = pd_mask.sum()
    cols = [
        "NP3TOT_first", "NP3TOT_last", "NP3TOT_change", "NP3TOT_change_rate",
        "NHY_first", "NHY_last", "NHY_change",
        "MOCA_first", "MOCA_last", "MOCA_change",
        "disease_duration_first",
    ]
    print(f"\nCoverage (PD subjects, n={n_pd}):")
    for col in cols:
        if col in df.columns:
            n = df.loc[pd_mask, col].notna().sum()
            print(f"  {col:<28} {n:>3} / {n_pd}")

    print(f"\nPDSTATE_first distribution (PD):")
    print(df.loc[pd_mask, "PDSTATE_first"].value_counts(dropna=False).to_string())

    print(f"\nNP3TOT_change_rate stats (PD):")
    print(df.loc[pd_mask, "NP3TOT_change_rate"].describe().round(2).to_string())

    print(f"\ndisease_duration_first stats (PD):")
    print(df.loc[pd_mask, "disease_duration_first"].describe().round(2).to_string())


if __name__ == "__main__":
    build_manifest_with_updrs()
