"""
generate_realistic.py
=====================
Generates a synthetic-realistic flood relief demand dataset
for Odisha and Bihar districts (2018–2023).

Run:
    python data/generate_realistic.py
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(42)

DISTRICTS = [
    # Odisha coastal flood-prone
    "Puri", "Kendrapara", "Jagatsinghpur", "Balasore",
    "Bhadrak", "Cuttack", "Jajpur", "Mayurbhanj",
    "Ganjam", "Khordha",
    # Bihar flood-prone
    "Darbhanga", "Sitamarhi", "Madhubani", "Supaul",
    "Saharsa", "Khagaria", "Katihar", "Muzaffarpur",
    "Bhagalpur", "Gopalganj"
]

POPULATION = {
    "Puri": 1497888, "Kendrapara": 1439891, "Jagatsinghpur": 1136605,
    "Balasore": 2317419, "Bhadrak": 1506522, "Cuttack": 2618708,
    "Jajpur": 1826275, "Mayurbhanj": 2513895, "Ganjam": 3520151,
    "Khordha": 2246341, "Darbhanga": 3921971, "Sitamarhi": 3419622,
    "Madhubani": 4476044, "Supaul": 2218876, "Saharsa": 1900661,
    "Khagaria": 1666886, "Katihar": 3071029, "Muzaffarpur": 4801062,
    "Bhagalpur": 3037766, "Gopalganj": 2562012,
}

RIVER_RISK = {
    "Puri": 0.8, "Kendrapara": 0.9, "Jagatsinghpur": 0.85, "Balasore": 0.75,
    "Bhadrak": 0.7, "Cuttack": 0.8, "Jajpur": 0.65, "Mayurbhanj": 0.5,
    "Ganjam": 0.7, "Khordha": 0.6, "Darbhanga": 0.95, "Sitamarhi": 0.9,
    "Madhubani": 0.85, "Supaul": 0.92, "Saharsa": 0.88, "Khagaria": 0.87,
    "Katihar": 0.75, "Muzaffarpur": 0.8, "Bhagalpur": 0.7, "Gopalganj": 0.82,
}

start_date = date(2018, 1, 1)
end_date   = date(2023, 12, 31)
delta      = timedelta(days=7)   # weekly records

records = []
current = start_date

while current <= end_date:
    month       = current.month
    week_of_year = current.isocalendar()[1]
    year        = current.year

    for district in DISTRICTS:
        pop = POPULATION[district]
        river_risk = RIVER_RISK[district]

        # Monsoon signal (June–October peak)
        monsoon_factor = np.clip(
            np.sin((month - 3) * np.pi / 7), 0, 1
        )

        # Rainfall
        base_rain = 5 + 200 * monsoon_factor
        rainfall_mm  = max(0, np.random.normal(base_rain, base_rain * 0.4))
        rain7d_mm    = rainfall_mm * 7  * np.random.uniform(0.6, 1.2)
        rain30d_mm   = rainfall_mm * 30 * np.random.uniform(0.5, 1.1)

        # Flood risk & severity
        flood_risk_score = np.clip(
            river_risk * monsoon_factor + np.random.normal(0, 0.05),
            0, 1
        )
        flood_severity = np.clip(
            flood_risk_score * np.random.uniform(0.7, 1.1),
            0, 1
        )

        # Affected population
        affected_fraction    = flood_severity ** 1.5 * np.random.uniform(0.01, 0.08)
        affected_population  = int(pop * affected_fraction)

        # SPHERE-based demand calculation with noise
        # Food: 2.1 kg/person/day × 7 days, packed in 5 kg packets
        food_packets   = int(affected_population * 0.3 * np.random.uniform(0.8, 1.2))
        # Water: 15 L/person/day × 7 days
        water_liters   = int(affected_population * 15 * 7 * np.random.uniform(0.85, 1.15))
        # ORS: 3 sachets/person for diarrhea risk
        ors_packets    = int(affected_population * 3  * flood_severity * np.random.uniform(0.9, 1.1))
        # Medical kits: 1 per 50 people
        medical_kits   = int(affected_population / 50 * np.random.uniform(0.8, 1.2))
        # Tarpaulins: 1 per 5 people displaced
        tarpaulins     = int(affected_population * 0.2 * np.random.uniform(0.85, 1.15))

        records.append({
            "date":               current.isoformat(),
            "year":               year,
            "month":              month,
            "week_of_year":       week_of_year,
            "district":           district,
            "state":              "Odisha" if district in list(POPULATION.keys())[:10] else "Bihar",
            "population":         pop,
            "rainfall_mm":        round(rainfall_mm, 1),
            "rainfall_7d_mm":     round(rain7d_mm, 1),
            "rainfall_30d_mm":    round(rain30d_mm, 1),
            "flood_risk_score":   round(flood_risk_score, 3),
            "flood_severity":     round(flood_severity, 3),
            "affected_population":affected_population,
            "demand_food_packets":  max(0, food_packets),
            "demand_water_liters":  max(0, water_liters),
            "demand_ors_packets":   max(0, ors_packets),
            "demand_medical_kits":  max(0, medical_kits),
            "demand_tarpaulins":    max(0, tarpaulins),
        })

    current += delta

df = pd.DataFrame(records)
output_path = "data/flood_relief_demand_realistic.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(f"Shape: {df.shape}")
print(df.head())
