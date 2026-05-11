import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2 { font-family: 'Space Mono', monospace; }
    .stApp { background-color: #0d0f1a; color: #e8eaf0; }

    .input-label { font-size: 13px; color: #7b82b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }

    .result-placed {
        background: linear-gradient(135deg, #1a2e1a, #1e3b1e);
        border: 2px solid #2d6b2d;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        font-size: 26px;
        font-family: 'Space Mono', monospace;
        color: #6fcf6f;
        margin-top: 20px;
    }
    .result-not-placed {
        background: linear-gradient(135deg, #2e1a1a, #3b1e1e);
        border: 2px solid #6b2d2d;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        font-size: 26px;
        font-family: 'Space Mono', monospace;
        color: #cf6f6f;
        margin-top: 20px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d6bff, #7ee8fa);
        color: #0d0f1a;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 0;
        width: 100%;
        font-size: 16px;
        margin-top: 10px;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] { background-color: #1a1d2e !important; color: #e8eaf0 !important; border-color: #2e3250 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Columns ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model   = joblib.load(os.path.join(base, "student_placement_model.pkl"))
    scaler  = joblib.load(os.path.join(base, "scaler_student.pkl"))
    columns = joblib.load(os.path.join(base, "student_coolumns.pkl"))
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Placement Predictor")
st.markdown("Enter the student's details below to predict their **placement status**.")
st.divider()

# ── Default ranges per column (sensible defaults) ─────────────────────────────
col_config = {
    "study_hours":           {"min": 0.0,  "max": 12.0, "default": 5.0,  "step": 0.5},
    "attendance":            {"min": 0.0,  "max": 100.0,"default": 75.0, "step": 1.0},
    "sleep_hours":           {"min": 3.0,  "max": 12.0, "default": 7.0,  "step": 0.5},
    "internet_usage":        {"min": 0.0,  "max": 12.0, "default": 4.0,  "step": 0.5},
    "assignments_completed": {"min": 0.0,  "max": 20.0, "default": 10.0, "step": 1.0},
    "previous_score":        {"min": 0.0,  "max": 100.0,"default": 60.0, "step": 1.0},
    "exam_score":            {"min": 0.0,  "max": 100.0,"default": 65.0, "step": 1.0},
}

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
input_values = {}

for i, col_name in enumerate(columns):
    cfg = col_config.get(col_name, {"min": 0.0, "max": 100.0, "default": 50.0, "step": 1.0})
    label = col_name.replace("_", " ").title()
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_values[col_name] = st.number_input(
            label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
        )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Placement"):
    input_df = pd.DataFrame([input_values])[columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    # Map prediction (0/1 or string)
    if prediction == 1 or str(prediction).lower() in ["placed", "yes", "1"]:
        st.markdown('<div class="result-placed">✅ &nbsp; PLACED</div>', unsafe_allow_html=True)
        st.success("This student is likely to get **placed**! 🎉")
    else:
        st.markdown('<div class="result-not-placed">❌ &nbsp; NOT PLACED</div>', unsafe_allow_html=True)
        st.warning("This student may **not get placed**. Consider improving key areas.")

    # Show input summary
    st.markdown("#### 📋 Input Summary")
    st.dataframe(
        pd.DataFrame([input_values]).rename(columns=lambda x: x.replace("_", " ").title()),
        use_container_width=True,
        hide_index=True,
    )