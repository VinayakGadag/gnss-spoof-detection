import joblib
import numpy as np


xgb = joblib.load("models/xgb_model.pkl")
lgb = joblib.load("models/lgb_model.pkl")


def ensemble_predict(X):

    p1 = xgb.predict_proba(X)[:,1]
    p2 = lgb.predict_proba(X)[:,1]

    return 0.6*p1 + 0.4*p2