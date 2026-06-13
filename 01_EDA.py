"""
01_EDA.py — Exploratory Data Analysis
======================================
Run:  python notebooks/01_EDA.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv("data/flood_relief_demand_realistic.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# ── Distribution of demand targets ───────────────────────
targets = [
    "demand_food_packets",
    "demand_water_liters",
    "demand_ors_packets",
    "demand_medical_kits",
    "demand_tarpaulins"
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Resource Demand Distributions", fontsize=14, fontweight="bold")

for ax, col in zip(axes, targets):
    df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col.replace("demand_", "").replace("_", " ").title())
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("reports/demand_distributions.png", dpi=150)
plt.show()
print("Saved: reports/demand_distributions.png")

# ── Correlation heatmap ───────────────────────────────────
numeric_cols = [
    "rainfall_mm", "rainfall_7d_mm", "rainfall_30d_mm",
    "flood_risk_score", "flood_severity", "affected_population",
    "demand_food_packets", "demand_water_liters",
    "demand_ors_packets", "demand_medical_kits", "demand_tarpaulins"
]

plt.figure(figsize=(12, 9))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=150)
plt.show()
print("Saved: reports/correlation_heatmap.png")

# ── Monthly demand trend ──────────────────────────────────
if "month" in df.columns:
    monthly = df.groupby("month")["demand_food_packets"].mean()
    plt.figure(figsize=(10, 5))
    monthly.plot(kind="bar", color="steelblue", edgecolor="white")
    plt.title("Average Food Packet Demand by Month", fontsize=13, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Avg Demand")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/monthly_demand.png", dpi=150)
    plt.show()
    print("Saved: reports/monthly_demand.png")

# ── Rainfall vs Demand scatter ────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(
    df["rainfall_mm"],
    df["demand_food_packets"],
    alpha=0.3,
    color="steelblue",
    s=10
)
plt.xlabel("Daily Rainfall (mm)")
plt.ylabel("Food Packet Demand")
plt.title("Rainfall vs Food Packet Demand", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/rainfall_vs_demand.png", dpi=150)
plt.show()
print("Saved: reports/rainfall_vs_demand.png")

print("\nEDA complete.")
