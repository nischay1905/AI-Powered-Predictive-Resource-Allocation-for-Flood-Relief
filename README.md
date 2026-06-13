#  AI-Powered Predictive Resource Allocation for Flood Relief

**Challenge 1.2 — NSS Open Projects | AI & Intelligent Systems Track**

---

##  Problem
Flood-prone districts in Odisha and Bihar receive relief supplies *after* disasters strike. This project uses machine learning to **forecast demand 7–30 days before floods** so NGOs can pre-position supplies.

---

##Streamlit App Link:- https://ai-powered-predictive-resource-allocation-for-flood-relief-gjb.streamlit.app/


##  Resources Predicted
| Resource | Unit |
|---|---|
| Food Packets | Packets |
| Drinking Water | Liters |
| ORS Packets | Sachets |
| Medical Kits | Kits |
| Tarpaulins | Sheets |

---

##  Dataset
- 20 districts across Odisha & Bihar
- 2018–2023 weekly records (~43,800 rows)
- Features: rainfall, flood severity, flood risk score, affected population

---

##  Model: LightGBM Regressor

| Metric | Value |
|---|---|
| MAE | ~210 |
| RMSE | ~368 |
| MAPE | ~14.8%  |
| R² | ~0.91 |

> Challenge target: MAPE < 20%  Achieved

---

##  Files
| File | Purpose |
|---|---|
| `flood_relief_demand_realistic.csv` | Main dataset |
| `generate_realistic.py` | Dataset generator |
| `train_model.py` | Train all 5 ML models |
| `predict.py` | Run a quick prediction |
| `app.py` | Streamlit dashboard |
| `01_EDA.py` | Exploratory data analysis |
| `02_Feature_Importance.py` | Feature importance charts |

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_model.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

### 4. Quick prediction
```bash
python predict.py
```

---

## 📋 NGO Action Playbook
| Zone | Flood Risk | Action |
|---|---|---|
|  Green | < 30% | Normal inventory |
|  Yellow | 30–60% | Pre-position 50% reserve stock |
|  Red | > 60% | Deploy emergency inventory now |

---

##  Tech Stack
Python · LightGBM · Scikit-Learn · Streamlit · Pandas · Matplotlib · Seaborn

---

##  References
- [SPHERE Humanitarian Standards](https://spherestandards.org/)
- [NDMA Flood Guidelines](https://ndma.gov.in/)
- [IMD Rainfall Data](https://mausam.imd.gov.in/)
