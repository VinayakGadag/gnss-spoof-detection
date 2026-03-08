# GNSS Anti-Spoofing Detection System

### Hackathon Submission — Kaizen × ARIES × NyneOS (IIT Delhi)

---

# 1. Project Overview

Global Navigation Satellite Systems (GNSS) such as GPS are critical infrastructure used in aviation, maritime navigation, telecommunications, drones, and financial systems. However, GNSS signals are inherently weak and broadcast openly, making them vulnerable to **spoofing attacks**.

GNSS spoofing occurs when an attacker transmits counterfeit satellite signals that cause receivers to compute incorrect position, velocity, or timing information.

The objective of this project is to **detect spoofed GNSS signals using machine learning and physics-inspired signal analysis**.

The system analyzes satellite signal characteristics and identifies inconsistencies that indicate potential spoofing attempts.

This solution was developed for the **GNSS Anti-Spoofing AI Hackathon organized by NyneOS Technologies in collaboration with Kaizen 2026, EES, and ARIES at IIT Delhi**.

---

# 2. Problem Understanding

Real GNSS signals follow strict **physical constraints** due to satellite orbital dynamics and receiver motion.

Spoofed signals may mimic real signals but often introduce detectable inconsistencies in:

• Doppler shift evolution
• Pseudorange consistency across satellites
• Carrier phase continuity
• Correlator output relationships
• Signal power patterns

These inconsistencies form the basis of spoof detection.

The task is therefore framed as a **binary classification problem**:

`
0 → Genuine GNSS signal  
1 → Spoofed GNSS signal
`

The model learns patterns from the training dataset and predicts spoofed signals in the test dataset.

---

# 3. System Architecture

The detection pipeline combines **signal feature engineering with machine learning models**.

`
GNSS Signal Dataset
        ↓
Data Preprocessing & Validation
        ↓
Feature Engineering
    • Temporal signal dynamics
    • Correlator signal relationships
    • Satellite geometry consistency
        ↓
Physics-Inspired Spoof Indicators
        ↓
Machine Learning Models
    • XGBoost
    • LightGBM
        ↓
Ensemble Prediction
        ↓
Timestamp-Level Aggregation
        ↓
Final Spoof Detection Output
`

This architecture combines **data-driven learning with domain knowledge from GNSS signal behavior**.

---

# 4. Repository Structure

`
GNSS/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── physics_checks.py
│   ├── train.py
│   ├── predict.py
│   ├── ensemble.py
│   ├── optimize.py
│   ├── threshold.py
│   └── feature_selection.py
│
├── models/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   └── feature_columns.pkl
│
├── outputs/
│   └── submission.csv
│
└── README.md
`

---

# 5. Dataset Overview

The dataset consists of GNSS receiver measurements for multiple satellites.

Key columns include:

| Column               | Description                                   |
| -------------------- | --------------------------------------------- |
| PRN                  | Satellite identifier                          |
| Carrier Doppler hz   | Frequency shift caused by satellite motion    |
| Pseudorange m        | Estimated distance from receiver to satellite |
| RX time              | Receiver timestamp                            |
| Carrier phase cycles | Signal phase measurement                      |
| EC / LC / PC         | Correlator outputs measuring signal alignment |
| PIP / PQP            | Signal quality metrics                        |
| TCD                  | Code delay measurement                        |
| CN0                  | Carrier-to-noise ratio                        |

The target variable indicates whether the signal snapshot is **spoofed or genuine**.

---

# 6. Feature Engineering Strategy

Feature engineering is critical because GNSS spoofing introduces **temporal and physical inconsistencies** in signals.

## 6.1 Temporal Features

Temporal evolution of signals is analyzed using satellite-wise time differences.

Examples:

• Doppler drift per satellite
• Carrier phase jumps
• Pseudorange changes over time
• Code delay variations

These capture abnormal signal transitions.

---

## 6.2 Signal Relationship Features

Correlator outputs measure signal alignment with expected satellite codes.

Features include:

• Correlator distortion (|EC − LC|)
• Correlator balance ratio
• Signal quality ratios (PIP / PQP)

Spoofed signals often distort these relationships.

---

## 6.3 Satellite Geometry Features

Consistency across satellites is evaluated at each receiver timestamp.

Examples:

• Pseudorange spread across satellites
• Doppler spread across satellites
• CN0 deviation from mean signal strength

These capture **geometric inconsistencies introduced by spoofing**.

---

## 6.4 Rolling Statistical Features

Rolling statistics capture short-term signal behavior.

Examples:

• Rolling mean of Doppler per satellite
• Rolling standard deviation of Doppler

These detect abnormal signal dynamics.

---

## 6.5 Physics-Inspired Spoof Indicators

The system also includes domain-driven features derived from GNSS physics.

These include:

• Doppler drift magnitude
• Pseudorange velocity anomalies
• Carrier phase discontinuities
• Correlator distortion metrics
• Satellite geometry spread

These signals are combined into a **physics-based spoof score** that helps the model identify suspicious patterns.

---

# 7. Machine Learning Models

Two gradient boosting models are trained:

### XGBoost Classifier

Strengths:

• strong performance on structured data
• ability to capture nonlinear feature interactions
• robustness to noisy features

---

### LightGBM Classifier

Strengths:

• fast training on large datasets
• efficient tree-based learning
• good generalization

---

### Ensemble Prediction

Predictions from both models are combined using weighted averaging.

Benefits:

• improved stability
• reduced model bias
• better generalization

---

# 8. Training Strategy

Training follows a robust evaluation procedure.

• Stratified K-Fold cross validation (k = 5)
• Balanced evaluation using **Weighted F1 Score**
• Consistent feature schema between training and inference

The use of stratified folds ensures fair evaluation under **class imbalance conditions**.

---

# 9. Running the Project

## 9.1 Open the project directory

`
cd GNSS
`

---

## 9.2 Activate virtual environment

Windows:

`
venv\Scripts\activate
`

---

## 9.3 Install dependencies

`
pip install pandas numpy scikit-learn xgboost lightgbm joblib shap optuna
`

---

## 9.4 Train the model

`
python src/train.py
`

Artifacts generated:

`
models/xgb_model.pkl
models/lgb_model.pkl
models/feature_columns.pkl
`

---

## 9.5 Generate predictions

`
python src/predict.py
`

Output file:

`
outputs/submission.csv
`

---

# 10. Submission Format

The final submission file contains predictions for the test dataset.

`
outputs/submission.csv
`

Columns:

| Column     | Description            |
| ---------- | ---------------------- |
| time       | Receiver timestamp     |
| Spoofed    | Predicted label        |
| Confidence | Model confidence score |

Predictions are aggregated at the **timestamp level** to stabilize multi-satellite observations.

---

# 11. Reproducibility

This project is designed for reproducible results.

Measures include:

• deterministic training seeds
• consistent feature schemas
• aligned preprocessing for train and test data
• modular pipeline design

This ensures that training and inference produce consistent outputs.

---

# 12. Robustness Considerations

The system is designed to generalize beyond the training dataset.

Robustness strategies include:

• physics-based features independent of dataset distribution
• ensemble modeling
• stratified cross validation
• timestamp-level aggregation of predictions

These help the system remain stable under varying signal conditions.

---

# 13. Future Extensions

Possible improvements include:

• temporal sequence models (LSTM / Transformer)
• signal spectrum analysis
• satellite consistency graphs
• model calibration for confidence estimation

The current architecture allows easy extension for future improvements.

---

# 14. Hackathon Compliance

This submission complies with the hackathon requirements:

✔ Uses only the official dataset
✔ Provides reproducible training pipeline
✔ Generates predictions for the test dataset
✔ Includes detailed documentation and methodology

---

# 15. Conclusion

This system combines **GNSS signal analysis, physics-inspired features, and ensemble machine learning models** to detect spoofing attacks.

By integrating domain knowledge with data-driven learning, the approach aims to provide a **robust and scalable solution for GNSS spoof detection in real-world environments**.

---
