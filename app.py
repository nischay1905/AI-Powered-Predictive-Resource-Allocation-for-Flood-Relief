import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Flood Relief Forecasting", page_icon="🌊", layout="wide")

FEATURES = [
    "rainfall_mm", "rainfall_7d_mm", "rainfall_30d_mm",
    "flood_risk_score", "flood_severity", "affected_population",
    "month", "week_of_year"
]

TARGETS = {
    "🍱 Food Packets":   "demand_food_packets",
    "💧 Water (Liters)": "demand_water_liters",
    "💊 ORS Packets":    "demand_ors_packets",
    "🏥 Medical Kits":   "demand_medical_kits",
    "⛺ Tarpaulins":     "demand_tarpaulins",
}

@st.cache_resource
def train_models():
    df = pd.read_csv("flood_relief_demand_realistic.csv")
    models = {}
    for label, target in TARGETS.items():
        X = df[FEATURES]
        y = df[target]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        models[label] = m
    return models

st.title("🌊 AI-Powered Flood Relief Resource Allocation")
st.markdown("Predict disaster relief demand before floods strike — helping NGOs pre-position supplies.")

with st.spinner("Loading models (first run takes ~30 seconds)..."):
    models = train_models()

st.divider()

st.sidebar.header("📍 Scenario Parameters")
district = st.sidebar.selectbox("District", [
    "Puri", "Kendrapara", "Jagatsinghpur", "Balasore", "Bhadrak",
    "Darbhanga", "Sitamarhi", "Madhubani", "Supaul", "Muzaffarpur"
])

st.sidebar.subheader("🌧️ Rainfall")
rainfall = st.sidebar.slider("Daily Rainfall (mm)", 0, 500, 100)
rain7    = st.sidebar.slider("7-Day Cumulative (mm)", 0, 1500, 300)
rain30   = st.sidebar.slider("30-Day Cumulative (mm)", 0, 4000, 800)

st.sidebar.subheader("⚠️ Flood Conditions")
risk     = st.sidebar.slider("Flood Risk Score", 0.0, 1.0, 0.65, step=0.01)
severity = st.sidebar.slider("Flood Severity", 0.0, 1.0, 0.50, step=0.01)
affected = st.sidebar.number_input("Affected Population", 100, 500000, 10000, step=500)

st.sidebar.subheader("📅 Time")
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=7,
    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
week = st.sidebar.slider("Week of Year", 1, 52, 30)

input_df = pd.DataFrame({
    "rainfall_mm": [rainfall], "rainfall_7d_mm": [rain7],
    "rainfall_30d_mm": [rain30], "flood_risk_score": [risk],
    "flood_severity": [severity], "affected_population": [affected],
    "month": [month], "week_of_year": [week]
})

if risk < 0.3:
    st.subheader("🟢 GREEN ZONE — Normal Operations")
elif risk < 0.6:
    st.subheader("🟡 YELLOW ZONE — Pre-position 50% Reserve Stock")
else:
    st.subheader("🔴 RED ZONE — Deploy Emergency Inventory Now")

st.progress(risk)
st.subheader(f"📦 Predicted Resource Demand — {district}")

cols = st.columns(5)
for col, (label, model) in zip(cols, models.items()):
    pred = int(model.predict(input_df)[0])
    col.metric(label, f"{pred:,}")

st.divider()
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

st.line_chart(pd.DataFrame(chart_data, index=severity_range.round(2)))

st.divider()
st.subheader("📋 NGO Action Playbook")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🟢 Green Zone\n**Risk < 30%**\n\n- Normal inventory\n- Review supply chain\n- Update contact lists")
with col2:
    st.markdown("### 🟡 Yellow Zone\n**Risk 30–60%**\n\n- Move 50% reserve stock\n- Alert field teams\n- Confirm transport routes")
with col3:
    st.markdown("### 🔴 Red Zone\n**Risk > 60%**\n\n- Deploy full emergency inventory\n- Activate relief camps\n- Coordinate with NDMA/SDMA")

st.divider()
st.caption("Model: LightGBM | Data: Odisha & Bihar 2018–2023 | Challenge 1.2 — NSS Open Projects")
