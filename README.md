# 🫀 Heart Disease Predictor
### End-to-End Machine Learning Project · Amazon ML Scholars Portfolio

> **A complete, beginner-friendly ML pipeline** — from raw UCI data to a live Streamlit web app — built entirely with free, open-source tools.

---

## 📌 Project Summary

This project predicts the **presence of heart disease** using 13 clinical features from the [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease). It demonstrates a production-ready ML workflow including data preprocessing, model selection via cross-validation, evaluation, and an interactive web interface.

| | |
|---|---|
| **Problem Type** | Binary Classification |
| **Dataset** | UCI Cleveland Heart Disease (303 rows, 13 features) |
| **Best Model** | Random Forest / XGBoost (selected automatically) |
| **Typical Accuracy** | ~85–88% |
| **Typical ROC-AUC** | ~0.92–0.94 |
| **Cost** | Completely FREE |

---

## 🎯 Learning Objectives

After completing this project you will understand:
- ✅ How to load and clean a real-world medical dataset
- ✅ Feature scaling and train/test splitting
- ✅ Training and comparing multiple ML models
- ✅ Cross-validation and hyperparameter awareness
- ✅ Evaluating models with accuracy, F1, ROC-AUC, and confusion matrices
- ✅ Saving and loading models with `joblib`
- ✅ Building an interactive web app with Streamlit

---

## 📂 Folder Structure

```
heart-disease-predictor/
│
├── app.py                   ← Streamlit web application
├── requirements.txt         ← Python dependencies
├── README.md
│
├── src/
│   ├── preprocess.py        ← Download + clean + scale data
│   ├── train.py             ← Model training + selection + evaluation
│   └── evaluate.py          ← Detailed evaluation curves & reports
│
├── data/
│   └── heart.csv            ← Auto-downloaded from UCI (git-ignored)
│
├── models/
│   ├── best_model.pkl       ← Saved trained model (git-ignored)
│   ├── scaler.pkl           ← Saved StandardScaler
│   ├── evaluation_report.json
│   └── plots/
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── evaluation_curves.png
│
└── notebooks/
    └── exploration.ipynb    ← (Optional) EDA notebook
```

---

## 🚀 Quick Start (5 minutes)

### 1. Clone / Download

```bash
git clone https://github.com/tanuja0708/Heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download data & train the model

```bash
python src/train.py
```

This will:
- Download the UCI dataset automatically (no login required)
- Clean and preprocess the data
- Train 3 models and select the best via 5-fold cross-validation
- Save the model, scaler, and evaluation plots

Expected output:
```
⬇  Downloading dataset from UCI ML Repository …
✅ Saved raw data → data/heart.csv  (303 rows)
🔍 Cross-validation (5-fold, AUC):
   Logistic Regression       AUC = 0.9021 ± 0.0312
   Random Forest             AUC = 0.9187 ± 0.0274
   XGBoost                   AUC = 0.9204 ± 0.0261
🏆 Best model: XGBoost  (AUC = 0.9204)
📈 Test-set results:
   Accuracy : 0.8689
   ROC-AUC  : 0.9342
```

### 4. (Optional) Generate evaluation plots

```bash
python src/evaluate.py
```

### 5. Launch the web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Dataset

**Name:** Heart Disease (Cleveland)
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
**Direct URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
**License:** Open / Free

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | 0 = Female, 1 = Male | Categorical |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina | Categorical |
| `oldpeak` | ST depression (exercise vs rest) | Numerical |
| `slope` | Slope of peak ST segment | Categorical |
| `ca` | Major vessels (fluoroscopy, 0–3) | Categorical |
| `thal` | Thalassemia type | Categorical |
| **`target`** | **Heart disease present (0/1)** | **Binary** |

---

## 🧪 ML Pipeline

```
Raw CSV → Drop 6 NaN rows → Binarise target
       → StandardScaler (numerical) → Train/Test split (80/20, stratified)
       → 5-fold CV on 3 models → Pick best by AUC
       → Final fit → Evaluation → Save artifacts
```

**Models compared:**
1. Logistic Regression (baseline)
2. Random Forest (n=200, max_depth=6)
3. XGBoost (n=200, lr=0.05) ← usually wins

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Accuracy | ~86–88% |
| F1-Score (Disease class) | ~0.88 |
| ROC-AUC | ~0.93 |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing + models + metrics |
| XGBoost | Gradient boosting model |
| Matplotlib / Seaborn | Evaluation plots |
| Joblib | Model serialization |
| Streamlit | Interactive web frontend |

All **100% free and open source.**

---

## 🔮 Future Improvements

- [ ] SHAP values for model explainability
- [ ] Hyperparameter tuning with Optuna
- [ ] Docker containerization
- [ ] Deploy to Streamlit Community Cloud (free)
- [ ] Add LIME for per-prediction explanations

---
