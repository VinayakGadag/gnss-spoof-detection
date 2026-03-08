import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb

from preprocess import load_data, preprocess
from features import build_features


TARGET = "target"


def train():
    print("Loading training data...")
    df = load_data("data/train.csv")

    print("Preprocessing data...")
    df = preprocess(df)

    print("Building features...")
    df = build_features(df)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = X.select_dtypes(include=["number", "bool"]).copy()
    X.columns = [col.replace(" ", "_") for col in X.columns]

    xgb_model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    lgb_model = lgb.LGBMClassifier(
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        verbosity=-1,
        force_row_wise=True,
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    print("Training model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)

        preds = xgb_model.predict(X_val)
        score = f1_score(y_val, preds, average="weighted")
        scores.append(score)
        print(f"Fold {fold} F1: {score:.4f}")

    print(f"Mean F1: {sum(scores) / len(scores):.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(lgb_model, "models/lgb_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    print("Models saved successfully.")


if __name__ == "__main__":
    train()
