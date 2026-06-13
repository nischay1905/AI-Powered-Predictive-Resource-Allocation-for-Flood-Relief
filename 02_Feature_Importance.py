"""
02_Feature_Importance.py — Feature Importance & Model Evaluation
================================================================
Run after train_model.py:  python notebooks/02_Feature_Importance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

FEATURES = [
    "rainfall_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "flood_risk_score",
    "flood_severity",
    "affected_population",
    "month",
    "week_of_year"
]

model_path = "models/food_packets_model.pkl"

if not os.path.exists(model_path):
    print("Model not found. Run src/train_model.py first.")
    exit()

model = joblib.load(model_path)

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(feat_df["Feature"], feat_df["Importance"], color="steelblue")
plt.xlabel("Feature Importance Score")
plt.title("LightGBM — Feature Importance\n(Food Packet Demand Model)",
          fontsize=13, fontweight="bold")
plt.tight_layout()

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/feature_importance.png", dpi=150)
plt.show()
print("Saved: reports/feature_importance.png")

# ── Scenario comparison ───────────────────────────────────
scenarios = pd.DataFrame({
    "Scenario":            ["Low Risk",  "Medium Risk", "High Risk",  "Extreme"],
    "rainfall_mm":         [20,          80,            200,          400],
    "rainfall_7d_mm":      [60,          250,           600,          1200],
    "rainfall_30d_mm":     [200,         700,           1800,         3500],
    "flood_risk_score":    [0.1,         0.4,           0.75,         0.95],
    "flood_severity":      [0.05,        0.35,          0.70,         0.90],
    "affected_population": [500,         5000,          25000,        80000],
    "month":               [3,           7,             8,            8],
    "week_of_year":        [10,          28,            32,           33],
})

feats   = scenarios[FEATURES]
preds   = model.predict(feats)

print("\nScenario Forecast — Food Packets")
print("=" * 45)
for i, row in scenarios.iterrows():
    print(f"{row['Scenario']:<15}: {int(preds[i]):>10,} packets")
print("=" * 45)
