import pandas as pd
import numpy as np

from physics_checks import compute_physics_scores


def temporal_features(df):

    df["doppler_drift"] = df.groupby("PRN")["Carrier Doppler hz"].diff()

    df["phase_jump"] = df.groupby("PRN")["Carrier phase cycles"].diff().abs()

    df["pseudo_jump"] = df.groupby("PRN")["Pseudorange m"].diff()

    df["tcd_jump"] = df.groupby("PRN")["TCD"].diff()

    return df


def signal_features(df):

    df["corr_error"] = np.abs(df["EC"] - df["LC"])

    df["corr_balance"] = (df["EC"] + df["LC"]) / (df["PC"] + 1e-6)

    df["pip_pqp_ratio"] = df["PIP"] / (df["PQP"] + 1e-6)

    return df


def satellite_features(df):

    df["pseudo_std"] = df.groupby("RX time")["Pseudorange m"].transform("std")

    df["doppler_std"] = df.groupby("RX time")["Carrier Doppler hz"].transform("std")

    df["cn0_mean"] = df.groupby("RX time")["CN0"].transform("mean")

    df["cn0_dev"] = df["CN0"] - df["cn0_mean"]

    return df


def statistical_features(df):

    window = 5

    df["doppler_roll_mean"] = (
        df.groupby("PRN")["Carrier Doppler hz"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["doppler_roll_std"] = (
        df.groupby("PRN")["Carrier Doppler hz"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def build_features(df):

    df = temporal_features(df)

    df = signal_features(df)

    df = satellite_features(df)

    df = statistical_features(df)

    df = compute_physics_scores(df)

    df = df.fillna(0)

    return df