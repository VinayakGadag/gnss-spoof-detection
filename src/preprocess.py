from pathlib import Path

import pandas as pd


def resolve_data_path(path):
    requested = Path(path)
    if requested.exists():
        return requested

    fallbacks = {
        "train.csv": "_train.csv",
        "sample_submission.csv": "submission_format.csv",
    }

    fallback_name = fallbacks.get(requested.name)
    if fallback_name:
        fallback = requested.with_name(fallback_name)
        if fallback.exists():
            return fallback

    tried = [str(requested)]
    if fallback_name:
        tried.append(str(requested.with_name(fallback_name)))
    raise FileNotFoundError(
        f"Could not find dataset file. Tried: {', '.join(tried)}"
    )


def load_data(path):
    return pd.read_csv(resolve_data_path(path), low_memory=False)

def preprocess(df):
    rename_map = {
        "Carrier_Doppler_hz": "Carrier Doppler hz",
        "Pseudorange_m": "Pseudorange m",
        "RX_time": "RX time",
        "Carrier_phase": "Carrier phase cycles",
        "spoofed": "target",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_cols = [
        "PRN",
        "Carrier Doppler hz",
        "Pseudorange m",
        "RX time",
        "TOW",
        "Carrier phase cycles",
        "EC",
        "LC",
        "PC",
        "PIP",
        "PQP",
        "TCD",
        "CN0",
        "target",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "target" in df.columns:
        df = df[df["target"].notna()].copy()
        df["target"] = df["target"].astype(int)

    return df
