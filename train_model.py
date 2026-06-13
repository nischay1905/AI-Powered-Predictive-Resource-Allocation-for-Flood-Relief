import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/flood_relief_demand_realistic.csv")

print(f"Dataset shape: {df.shape}")
print(df.head())

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

TARGETS = [
    "demand_food_packets",
    "demand_water_liters",
    "demand_ors_packets",
    "demand_medical_kits",
    "demand_tarpaulins"
]

results = {}

for target in TARGETS:
    print(f"\nTraining model for: {target}")

    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100

    results[target] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }

    model_name = target.replace("demand_", "") + "_model.pkl"
    joblib.dump(model, f"models/{model_name}")
    print(f"  Saved: models/{model_name}")

print("\n" + "=" * 55)
print(f"{'Target':<25} {'MAE':>7} {'RMSE':>8} {'MAPE':>7} {'R2':>7}")
print("=" * 55)
for target, m in results.items():
    label = target.replace("demand_", "")
    print(f"{label:<25} {m['MAE']:>7} {m['RMSE']:>8} {m['MAPE']:>6}% {m['R2']:>7}")
print("=" * 55)

print("\nAll models saved successfully.")
