import shap
import joblib
import pandas as pd
from preprocess import load_data

model = joblib.load("models/xgb_model.pkl")

df = load_data("data/train.csv")

X = df.drop(columns=["target"])

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

importance = abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    "feature":X.columns,
    "importance":importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

top_features = feature_importance.head(25)["feature"].tolist()

print(top_features)
