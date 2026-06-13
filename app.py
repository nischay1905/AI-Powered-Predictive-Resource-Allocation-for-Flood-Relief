import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Flood Relief Forecasting Dashboard",
    page_icon="🌊",
    layout="wide"
)

# ── Load Models ───────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_models():
    models = {}
    model_map = {
        "Food Packets":   "food_packets_model.pkl",
        "Water (Liters)": "water_liters_model.pkl",
        "ORS Packets":    "ors_packets_model.pkl",
        "Medical Kits":   "medical_kits_model.pkl",
        "Tarpaulins":     "tarpaulins_model.pkl",
    }
    for label, fname in model_map.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[label] = joblib.load(path)
    return models

models = load_models()

# ── Header ────────────────────────────────────────────────
st.title("🌊 AI-Powered Flood Relief Resource Allocation")
st.markdown(
    "Predict disaster relief demand 7–30 days ahead "
    "to help NGOs pre-position supplies before floods strike."
)
st.divider()

# ── Sidebar Inputs ────────────────────────────────────────
st.sidebar.header("📍 Scenario Parameters")

district = st.sidebar.selectbox(
    "District",
    [
        "Puri", "Kendrapara", "Jagatsinghpur", "Balasore",
        "Bhadrak", "Darbhanga", "Sitamarhi", "Madhubani",
        "Supaul", "Saharsa", "Khagaria", "Katihar",
        "Muzaffarpur", "Bhagalpur", "Gopalganj"
    ]
)

st.sidebar.subheader("🌧️ Rainfall")
rainfall    = st.sidebar.slider("Daily Rainfall (mm)",      0,   500,  100)
rain7       = st.sidebar.slider("7-Day Cumulative (mm)",    0,  1500,  300)
rain30      = st.sidebar.slider("30-Day Cumulative (mm)",   0,  4000,  800)

st.sidebar.subheader("⚠️ Flood Conditions")
risk        = st.sidebar.slider("Flood Risk Score",  0.0, 1.0, 0.65, step=0.01)
severity    = st.sidebar.slider("Flood Severity",    0.0, 1.0, 0.50, step=0.01)
affected    = st.sidebar.number_input("Affected Population", 100, 500_000, 10_000, step=500)

st.sidebar.subheader("📅 Time")
month_names = {
    1:"January",2:"February",3:"March",4:"April",
    5:"May",6:"June",7:"July",8:"August",
    9:"September",10:"October",11:"November",12:"December"
}
month       = st.sidebar.selectbox("Month", list(range(1,13)), index=7,
                                   format_func=lambda x: month_names[x])
week        = st.sidebar.slider("Week of Year", 1, 52, 30)

# ── Build Input ───────────────────────────────────────────
input_df = pd.DataFrame({
    "rainfall_mm":        [rainfall],
    "rainfall_7d_mm":     [rain7],
    "rainfall_30d_mm":    [rain30],
    "flood_risk_score":   [risk],
    "flood_severity":     [severity],
    "affected_population":[affected],
    "month":              [month],
    "week_of_year":       [week]
})

# ── Risk Badge ────────────────────────────────────────────
def risk_zone(score):
    if score < 0.3:
        return "🟢 GREEN — Normal Operations", "green"
    elif score < 0.6:
        return "🟡 YELLOW — Pre-position 50% Reserve", "orange"
    else:
        return "🔴 RED — Deploy Emergency Inventory Now", "red"

zone_label, zone_color = risk_zone(risk)
st.subheader(f"Flood Risk Zone: {zone_label}")
st.progress(risk)
st.divider()

# ── Predictions ───────────────────────────────────────────
st.subheader(f"📦 Predicted Resource Demand — {district}")

if not models:
    st.error(
        "No trained models found. "
        "Run `python src/train_model.py` from the project root first."
    )
else:
    cols = st.columns(len(models))
    icons = ["🍱", "💧", "💊", "🏥", "⛺"]

    for col, (label, model), icon in zip(cols, models.items(), icons):
        pred = int(model.predict(input_df)[0])
        col.metric(f"{icon} {label}", f"{pred:,}")

    st.divider()

    # ── Sensitivity: what happens at higher severity ──────
    st.subheader("📈 Demand vs Flood Severity")
    severity_range = np.linspace(0, 1, 50)
    chart_data = {}

    for label, model in models.items():
        preds = []
        for s in severity_range:
            row = input_df.copy()
            row["flood_severity"] = s
            preds.append(int(model.predict(row)[0]))
        chart_data[label] = preds

    chart_df = pd.DataFrame(chart_data, index=severity_range.round(2))
    chart_df.index.name = "Flood Severity"
    st.line_chart(chart_df)

    st.divider()

    # ── Playbook ──────────────────────────────────────────
    st.subheader("📋 NGO Action Playbook")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🟢 Green Zone\n**Flood Risk < 30%**\n\n"
                    "- Maintain normal inventory\n"
                    "- Review supply chain\n"
                    "- Update contact lists")
    with col2:
        st.markdown("### 🟡 Yellow Zone\n**Flood Risk 30–60%**\n\n"
                    "- Move 50% of reserve stock\n"
                    "- Alert field teams\n"
                    "- Confirm transport routes")
    with col3:
        st.markdown("### 🔴 Red Zone\n**Flood Risk > 60%**\n\n"
                    "- Deploy full emergency inventory\n"
                    "- Activate relief camps\n"
                    "- Coordinate with NDMA/SDMA")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: LightGBM Regressor | "
    "Data: Odisha & Bihar districts 2018–2023 | "
    "Challenge 1.2 — NSS Open Projects"
)
