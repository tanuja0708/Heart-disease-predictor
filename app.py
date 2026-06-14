"""
app.py — Streamlit frontend for the Heart Disease Prediction project.
Run with:  streamlit run app.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & palette ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    color: #f1f5f9;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.8rem !important; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 1.8rem !important; }

/* Prediction banner */
.pred-danger {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-left: 5px solid #ef4444;
    border-radius: 8px; padding: 20px 24px; margin: 16px 0;
}
.pred-safe {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 5px solid #22c55e;
    border-radius: 8px; padding: 20px 24px; margin: 16px 0;
}
.pred-danger h2 { color: #dc2626; margin: 0 0 4px 0; }
.pred-safe  h2 { color: #16a34a; margin: 0 0 4px 0; }
.pred-danger p, .pred-safe p { color: #374151; margin: 0; font-size: 0.95rem; }

/* Feature table */
.feat-table { font-size: 0.88rem; }

/* Section headers */
.section-label {
    font-size: 0.7rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & report ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/evaluation_report.json") as f:
        report = json.load(f)
    return model, scaler, report


def check_artifacts():
    missing = [p for p in ["models/best_model.pkl", "models/scaler.pkl",
                            "models/evaluation_report.json"]
               if not os.path.exists(p)]
    return missing


NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]

FEATURE_INFO = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of Peak ST Segment",
    "ca":       "Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia",
}


# ── Sidebar — input form ─────────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown("## 🫀 Patient Data")
    st.sidebar.markdown("---")

    age      = st.sidebar.slider("Age", 20, 80, 54)
    sex      = st.sidebar.selectbox("Sex", ["Female (0)", "Male (1)"])
    cp       = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina (0)", "Atypical Angina (1)",
         "Non-Anginal Pain (2)", "Asymptomatic (3)"]
    )
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol     = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
    restecg  = st.sidebar.selectbox(
        "Resting ECG Result",
        ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"]
    )
    thalach  = st.sidebar.slider("Max Heart Rate Achieved", 60, 210, 150)
    exang    = st.sidebar.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
    oldpeak  = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope    = st.sidebar.selectbox(
        "Slope of Peak ST Segment",
        ["Upsloping (0)", "Flat (1)", "Downsloping (2)"]
    )
    ca       = st.sidebar.selectbox("Major Vessels (Fluoroscopy)", ["0", "1", "2", "3"])
    thal     = st.sidebar.selectbox(
        "Thalassemia",
        ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Unknown (3)"]
    )

    # Extract the integer from each selectbox value
    def extract_int(s):
        return int(s.split("(")[-1].replace(")", ""))

    return {
        "age":      age,
        "sex":      extract_int(sex),
        "cp":       extract_int(cp),
        "trestbps": trestbps,
        "chol":     chol,
        "fbs":      extract_int(fbs),
        "restecg":  extract_int(restecg),
        "thalach":  thalach,
        "exang":    extract_int(exang),
        "oldpeak":  oldpeak,
        "slope":    extract_int(slope),
        "ca":       int(ca),
        "thal":     extract_int(thal),
    }


# ── Probability gauge (matplotlib) ───────────────────────────────────────────
def draw_gauge(prob: float):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta = np.linspace(np.pi, 0, 300)
    # Background arc
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color="#e2e8f0", solid_capstyle="round")
    # Filled arc
    filled = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    color  = "#ef4444" if prob > 0.5 else "#22c55e"
    ax.plot(np.cos(filled), np.sin(filled), lw=18, color=color, solid_capstyle="round")

    ax.text(0, 0.08, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.22, "Risk Score", ha="center", va="center",
            fontsize=11, color="#64748b")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='margin-bottom:8px'>
        <span style='font-size:2.4rem;'>🫀</span>
        <span style='font-size:2rem; font-weight:700; margin-left:10px; vertical-align:middle;'>
            Heart Disease Predictor
        </span>
    </div>
    <p style='color:#64748b; margin-top:-4px; font-size:1rem;'>
        ML-powered risk assessment · UCI Cleveland Heart Disease Dataset · Built for Amazon ML Scholars
    </p>
    <hr style='border:none; border-top:1px solid #e2e8f0; margin: 16px 0 24px;'>
    """, unsafe_allow_html=True)

    missing = check_artifacts()
    if missing:
        st.error(f"⚠️  Model not trained yet. Missing: {missing}\n\n"
                 "Run `python src/train.py` first.")
        st.code("python src/train.py", language="bash")
        return

    model, scaler, report = load_artifacts()
    inputs = sidebar_inputs()

    # ── Top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", report["best_model"].split()[0])
    col2.metric("Test Accuracy", f"{report['test_accuracy']*100:.1f}%")
    col3.metric("ROC-AUC Score", f"{report['test_roc_auc']:.3f}")
    col4.metric("Dataset", "Cleveland UCI")

    st.markdown("---")

    # ── Prediction ────────────────────────────────────────────────────────────
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("### 🔍 Prediction")

        raw_df = pd.DataFrame([inputs])
        scaled_df = raw_df.copy()
        scaled_df[NUMERICAL] = scaler.transform(scaled_df[NUMERICAL])

        pred   = model.predict(scaled_df)[0]
        prob   = model.predict_proba(scaled_df)[0][1]

        if pred == 1:
            st.markdown(f"""
            <div class='pred-danger'>
                <h2>⚠️  High Risk Detected</h2>
                <p>The model predicts <strong>elevated risk of heart disease</strong>
                   with <strong>{prob*100:.1f}% confidence</strong>.
                   Please consult a cardiologist for a professional assessment.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-safe'>
                <h2>✅  Low Risk</h2>
                <p>The model predicts <strong>low risk of heart disease</strong>
                   with <strong>{(1-prob)*100:.1f}% confidence</strong>.
                   Continue maintaining a heart-healthy lifestyle.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("**ℹ️  Disclaimer:** This tool is for educational purposes only and is not a medical diagnosis.")

        # Input summary table
        with st.expander("📋 View entered patient data"):
            readable = {FEATURE_INFO.get(k, k): v for k, v in inputs.items()}
            st.dataframe(
                pd.DataFrame(readable.items(), columns=["Feature", "Value"]),
                use_container_width=True, hide_index=True
            )

    with right:
        st.markdown("### 📊 Risk Gauge")
        fig = draw_gauge(prob)
        st.pyplot(fig, use_container_width=False)

        # Probability bar
        st.markdown(f"""
        <div style='margin-top:8px;'>
            <div class='section-label'>Disease Probability Breakdown</div>
            <div style='display:flex; gap:8px; align-items:center; font-size:0.9rem;'>
                <span style='color:#22c55e;'>No Disease: {(1-prob)*100:.1f}%</span>
                <span style='color:#94a3b8;'>|</span>
                <span style='color:#ef4444;'>Disease: {prob*100:.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── About + Model Performance ─────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📚 About the Dataset", "📈 Model Performance", "🚀 How It Works"])

    with tab1:
        st.markdown("""
        ### Cleveland Heart Disease Dataset
        | Property | Value |
        |----------|-------|
        | **Source** | UCI Machine Learning Repository (FREE) |
        | **URL** | https://archive.ics.uci.edu/ml/datasets/heart+disease |
        | **Rows** | 303 patients |
        | **Features** | 13 clinical features |
        | **Target** | Presence of heart disease (binary: 0/1) |
        | **License** | Open / Public domain |

        **Key Features Explained:**
        - `cp` — Chest pain type (most predictive feature)
        - `thal` — Thalassemia blood disorder type
        - `ca` — Number of major vessels colored by fluoroscopy
        - `oldpeak` — ST depression induced by exercise relative to rest
        """)

    with tab2:
        cm_path = "models/plots/confusion_matrix.png"
        ev_path = "models/plots/evaluation_curves.png"
        fi_path = "models/plots/feature_importance.png"

        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
        with c2:
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importances", use_column_width=True)
        if os.path.exists(ev_path):
            st.image(ev_path, caption="ROC & Precision-Recall Curves", use_column_width=True)

        st.markdown("**Cross-Validation AUC Scores (5-fold):**")
        cv_df = pd.DataFrame(
            report["cv_auc_scores"].items(),
            columns=["Model", "CV AUC"]
        ).sort_values("CV AUC", ascending=False)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("""
        ### Pipeline Overview
        ```
        Raw Data (UCI)
             │
             ▼
        Data Cleaning  →  Drop 6 missing rows, binarise target
             │
             ▼
        Preprocessing  →  StandardScaler on numerical features
             │
             ▼
        Model Selection →  5-fold CV on 3 candidate models
             │
             ▼
        Best Model Saved → models/best_model.pkl
             │
             ▼
        Streamlit App   →  Real-time inference (this page)
        ```

        **Stack:** Python · Scikit-learn · Pandas · NumPy · Matplotlib · Streamlit
        """)


if __name__ == "__main__":
    main()
