import numpy as np
import pandas as pd


def doppler_drift(df):

    return df.groupby("PRN")["Carrier Doppler hz"].diff()


def pseudorange_velocity(df):

    pr_diff = df.groupby("PRN")["Pseudorange m"].diff()
    time_diff = df.groupby("PRN")["RX time"].diff()

    velocity = pr_diff / (time_diff + 1e-6)

    return velocity


def phase_jump(df):

    return df.groupby("PRN")["Carrier phase cycles"].diff().abs()


def correlator_distortion(df):

    return np.abs(df["EC"] - df["LC"]) / (df["PC"] + 1e-6)


def geometry_spread(df):

    return df.groupby("RX time")["Pseudorange m"].transform("std")


def compute_physics_scores(df):

    df["physics_doppler"] = doppler_drift(df)
    df["physics_velocity"] = pseudorange_velocity(df)
    df["physics_phase"] = phase_jump(df)
    df["physics_corr"] = correlator_distortion(df)
    df["physics_geometry"] = geometry_spread(df)

    df["physics_spoof_score"] = (
        0.25 * df["physics_doppler"].abs()
        + 0.25 * df["physics_velocity"].abs()
        + 0.20 * df["physics_phase"]
        + 0.15 * df["physics_corr"]
        + 0.15 * df["physics_geometry"]
    )

    return df