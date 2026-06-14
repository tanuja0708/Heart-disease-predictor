"""
preprocess.py — Download and prepare the Heart Disease dataset from UCI ML Repository.
Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease (FREE, no login required)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Column names as defined by UCI ──────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Feature groups (useful for display)
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL   = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# ── Human-readable label maps ────────────────────────────────────────────────
LABEL_MAPS = {
    "sex":     {0: "Female", 1: "Male"},
    "cp":      {0: "Typical Angina", 1: "Atypical Angina",
                2: "Non-Anginal Pain", 3: "Asymptomatic"},
    "fbs":     {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
    "exang":   {0: "No", 1: "Yes"},
    "slope":   {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca":      {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal":    {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"},
}


def download_data(save_path: str = "data/heart.csv") -> pd.DataFrame:
    """
    Download the Cleveland Heart Disease dataset directly from UCI.
    Returns a clean DataFrame and saves it to save_path.
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/heart-disease/processed.cleveland.data"
    )
    print(f"⬇  Downloading dataset from UCI ML Repository …")
    df = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved raw data → {save_path}  ({len(df)} rows)")
    return df


def preprocess(df: pd.DataFrame, scaler_path: str = "models/scaler.pkl"):
    """
    Clean, encode, scale, and split the dataset.
    Returns X_train, X_test, y_train, y_test + feature names.
    """
    df = df.copy()

    # Binarise target  (0 = No Disease, 1 = Disease)
    df["target"] = (df["target"] > 0).astype(int)

    # Drop rows with missing values (only 6 rows in Cleveland dataset)
    df.dropna(inplace=True)

    print(f"\n📊 Class distribution after cleaning:")
    print(df["target"].value_counts().rename({0: "No Disease", 1: "Disease"}))

    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = X.columns.tolist()

    # Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features only
    scaler = StandardScaler()
    X_train[NUMERICAL] = scaler.fit_transform(X_train[NUMERICAL])
    X_test[NUMERICAL]  = scaler.transform(X_test[NUMERICAL])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved → {scaler_path}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    df = download_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)
    print(f"\n✅ Preprocessing complete.")
    print(f"   Train size : {X_train.shape}")
    print(f"   Test size  : {X_test.shape}")
    print(f"   Features   : {features}")
