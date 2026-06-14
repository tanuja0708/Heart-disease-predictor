"""
train.py — Train, tune, and save the best Heart Disease prediction model.
Compares Logistic Regression, Random Forest, and XGBoost; picks best by CV AUC.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import download_data, preprocess, NUMERICAL


# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/heart.csv"
MODEL_PATH  = "models/best_model.pkl"
REPORT_PATH = "models/evaluation_report.json"
PLOTS_DIR   = "models/plots"
RANDOM_SEED = 42

CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_leaf=3, random_state=RANDOM_SEED
    ),
}

# Try to add XGBoost if installed
try:
    from xgboost import XGBClassifier
    CANDIDATES["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_SEED
    )
except ImportError:
    print("ℹ  XGBoost not installed; skipping it (pip install xgboost to include).")


# ── Helpers ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    """Works for tree-based models that expose feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], color="#2563eb")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9
    )
    ax.set_title("Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main training routine ─────────────────────────────────────────────────────
def train():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        df = download_data(DATA_PATH)
    else:
        print(f"📂 Loading cached data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # ── 2. Model selection via cross-validation ───────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    print("\n🔍 Cross-validation (5-fold, AUC):")
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = scores.mean()
        print(f"   {name:25s}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=cv_results.get)
    best_model = CANDIDATES[best_name]
    print(f"\n🏆 Best model: {best_name}  (AUC = {cv_results[best_name]:.4f})")

    # ── 3. Final training on full train set ───────────────────────────────────
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # ── 4. Evaluation on held-out test set ───────────────────────────────────
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])

    print(f"\n📈 Test-set results:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")

    # ── 5. Save evaluation report (JSON) ─────────────────────────────────────
    eval_report = {
        "best_model"       : best_name,
        "cv_auc_scores"    : {k: round(v, 4) for k, v in cv_results.items()},
        "test_accuracy"    : round(acc, 4),
        "test_roc_auc"     : round(auc, 4),
        "confusion_matrix" : cm.tolist(),
        "classification_report": report,
        "feature_names"    : feature_names,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"📄 Report saved → {REPORT_PATH}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, f"Confusion Matrix — {best_name}", cm_path)

    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plot_feature_importance(best_model, feature_names, fi_path)

    print(f"\n✅ All done!  Plots → {PLOTS_DIR}/")
    return best_model, eval_report


if __name__ == "__main__":
    train()
