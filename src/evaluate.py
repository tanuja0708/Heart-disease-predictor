"""
evaluate.py — Load the saved model and generate a full evaluation report.
Run this at any time after training to re-check model performance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import download_data, preprocess

DATA_PATH   = "data/heart.csv"
MODEL_PATH  = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
PLOTS_DIR   = "models/plots"


def evaluate():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else download_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)

    # Load model
    model = joblib.load(MODEL_PATH)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Metrics summary ───────────────────────────────────────────────────────
    print("=" * 55)
    print("  EVALUATION REPORT")
    print("=" * 55)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#2563eb", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # ── Precision-Recall Curve ────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    axes[1].plot(recall, precision, color="#16a34a", lw=2,
                 label=f"PR curve (AUC = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    plt.suptitle("Model Evaluation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "evaluation_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n📊 Evaluation curves saved → {path}")


if __name__ == "__main__":
    evaluate()
