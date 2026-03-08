import sys
sys.path.append('..')

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb

from src.preprocess import load_data, preprocess
from src.features import build_features

TARGET = "target"


df = load_data("data/train.csv")
df = preprocess(df)
df = build_features(df)

y = df[TARGET]
X = df.drop(columns=[TARGET])


def objective(trial):

    model_type = trial.suggest_categorical("model", ["xgb", "lgb"])

    if model_type == "xgb":

        model = XGBClassifier(
            max_depth=trial.suggest_int("max_depth",4,10),
            learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
            n_estimators=trial.suggest_int("n_estimators",200,800),
            subsample=trial.suggest_float("subsample",0.6,1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
            gamma=trial.suggest_float("gamma",0,5),
            eval_metric="logloss",
            tree_method="hist"
        )

    else:

        model = lgb.LGBMClassifier(
            num_leaves=trial.suggest_int("num_leaves",20,200),
            learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
            n_estimators=trial.suggest_int("n_estimators",200,800),
            feature_fraction=trial.suggest_float("feature_fraction",0.6,1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction",0.6,1.0),
            bagging_freq=1
        )

    kf = StratifiedKFold(n_splits=5)

    scores = []

    for train_idx, val_idx in kf.split(X,y):

        X_train,X_val = X.iloc[train_idx],X.iloc[val_idx]
        y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]

        model.fit(X_train,y_train)

        pred = model.predict(X_val)

        score = f1_score(y_val,pred,average="weighted")

        scores.append(score)

    return sum(scores)/len(scores)


study = optuna.create_study(direction="maximize")

study.optimize(objective,n_trials=50)

print(study.best_params)