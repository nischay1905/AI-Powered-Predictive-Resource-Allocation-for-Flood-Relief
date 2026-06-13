import pandas as pd
import joblib

# ── Sample Input ──────────────────────────────────────────
sample = pd.DataFrame({
    "rainfall_mm":        [120],
    "rainfall_7d_mm":     [400],
    "rainfall_30d_mm":    [1000],
    "flood_risk_score":   [0.85],
    "flood_severity":     [0.80],
    "affected_population":[20000],
    "month":              [8],
    "week_of_year":       [32]
})

# ── Resources to predict ──────────────────────────────────
model_files = {
    "Food Packets":   "models/food_packets_model.pkl",
    "Water (Liters)": "models/water_liters_model.pkl",
    "ORS Packets":    "models/ors_packets_model.pkl",
    "Medical Kits":   "models/medical_kits_model.pkl",
    "Tarpaulins":     "models/tarpaulins_model.pkl",
}

print("\nFlood Relief Resource Forecast")
print("=" * 40)
for label, path in model_files.items():
    try:
        model = joblib.load(path)
        pred  = model.predict(sample)[0]
        print(f"{label:<20}: {int(pred):>8,}")
    except FileNotFoundError:
        print(f"{label:<20}: model not found — run train_model.py first")
print("=" * 40)
