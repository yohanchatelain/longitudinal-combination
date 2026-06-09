from __future__ import annotations

import pandas as pd


def _days_col(df: pd.DataFrame) -> str | None:
    if "days" in df.columns:
        return "days"
    if "days_since_enrollment" in df.columns:
        return "days_since_enrollment"
    return None


def _sort_cols(df: pd.DataFrame) -> list[str]:
    if "timepoint" in df.columns:
        return ["timepoint", "age"]
    return ["age"]


def _pair_record(t0, t1, index_t0, index_t1, days_col: str | None) -> dict:
    days_t0 = float(t0[days_col]) if days_col and pd.notna(t0[days_col]) else float("nan")
    days_t1 = float(t1[days_col]) if days_col and pd.notna(t1[days_col]) else float("nan")
    return {
        "id": t0["id"],
        "participant_id": t0["participant_id"],
        "directory_t0": t0["directory"],
        "directory_t1": t1["directory"],
        "age_t0": float(t0["age"]),
        "age_t1": float(t1["age"]),
        "delta_age": float(t1["age"]) - float(t0["age"]),
        "days_t0": days_t0,
        "days_t1": days_t1,
        "delta_days": days_t1 - days_t0,
        "sex": t0["sex"],
        "session_t0": t0["session"],
        "session_t1": t1["session"],
        "session_type_t0": t0.get("session_type", ""),
        "session_type_t1": t1.get("session_type", ""),
        "index_t0": int(index_t0),
        "index_t1": int(index_t1),
    }


def make_first_last_pairs(df: pd.DataFrame) -> pd.DataFrame:
    days_col = _days_col(df)
    records = []
    for _, group in df.sort_values(_sort_cols(df)).groupby("id", sort=False):
        if len(group) < 2:
            continue
        t0 = group.iloc[0]
        t1 = group.iloc[-1]
        records.append(_pair_record(t0, t1, group.index[0], group.index[-1], days_col))
    pairs = pd.DataFrame.from_records(records)
    if not pairs.empty and (pairs["age_t1"] < pairs["age_t0"]).any():
        print("Warning: age_t1 < age_t0 found in first-last pairs")
    return pairs


def make_all_forward_pairs(df: pd.DataFrame) -> pd.DataFrame:
    days_col = _days_col(df)
    records = []
    sort_cols = _sort_cols(df)
    for _, group in df.sort_values(sort_cols).groupby("id", sort=False):
        rows = list(group.iterrows())
        for left_pos, (idx0, t0) in enumerate(rows[:-1]):
            for idx1, t1 in rows[left_pos + 1 :]:
                if "timepoint" in df.columns and not (t1["timepoint"] > t0["timepoint"]):
                    continue
                records.append(_pair_record(t0, t1, idx0, idx1, days_col))
    pairs = pd.DataFrame.from_records(records)
    if not pairs.empty:
        if (pairs["age_t1"] < pairs["age_t0"]).any():
            print("Warning: age_t1 < age_t0 found in forward pairs")
        if "delta_days" in pairs and pairs["delta_days"].notna().any():
            bad = pairs["delta_days"].notna() & (pairs["delta_days"] <= 0)
            if bad.any():
                print(f"Warning: {int(bad.sum())} forward pairs have delta_days <= 0")
    return pairs
