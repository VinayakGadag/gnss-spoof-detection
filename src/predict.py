import sys
sys.path.append('..')

import joblib
import pandas as pd

from preprocess import load_data, preprocess
from features import build_features
from ensemble import ensemble_predict


def predict():

    df = load_data("data/test.csv")
    row_time = pd.to_numeric(df["time"], errors="coerce")

    df = preprocess(df)

    df = build_features(df)
    df = df.select_dtypes(include=["number", "bool"]).copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]

    feature_columns = joblib.load("models/feature_columns.pkl")
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    probs = ensemble_predict(df)

    threshold = 0.45

    by_time = (
        pd.DataFrame({"time": row_time, "prob": probs})
        .groupby("time", as_index=False)["prob"]
        .mean()
    )
    by_time["Spoofed"] = (by_time["prob"] > threshold).astype(int)
    by_time["Confidence"] = by_time["prob"]

    submission = load_data("data/sample_submission.csv")
    submission["time"] = pd.to_numeric(submission["time"], errors="coerce")
    submission = submission[["time"]].merge(
        by_time[["time", "Spoofed", "Confidence"]],
        on="time",
        how="left",
    )
    submission["Spoofed"] = submission["Spoofed"].fillna(0).astype(int)
    submission["Confidence"] = submission["Confidence"].fillna(0.0)

    submission.to_csv("outputs/submission.csv", index=False)

    print("Submission saved.")


if __name__ == "__main__":
    predict()
