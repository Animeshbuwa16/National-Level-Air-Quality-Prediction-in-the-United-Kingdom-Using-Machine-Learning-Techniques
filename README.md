# National-Level-Air-Quality-Prediction-in-the-United-Kingdom-Using-Machine-Learning-Techniques


1  UK National Air Quality Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-99.5%25_Accuracy-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-OGL_v3-lightgrey)
![University](https://img.shields.io/badge/Coventry_University-MSc_Data_Science-blue)

A machine learning system for predicting Air Quality Index (AQI)
categories across eleven major UK cities, developed as part of the
MSc Data Science dissertation at Coventry University (7150CEM).

---

2  Project Summary

| | |
|---|---|
| **Project** | UK National Air Quality Prediction |
| **Author** | Animesh Buwa |
| **Student ID** | 15863968 |
| **Institution** | Coventry University |
| **Module** | 7150CEM MSc Data Science Project |
| **Supervisor** | Omid Chatrabgoun |
| **Academic Year** | 2025/26 |
| **Models Used** | Random Forest, XGBoost, LSTM |
| **Best Accuracy** | XGBoost — 99.5% |
| **Cities Covered** | 11 major UK cities |
| **Data Period** | January 2021 – February 2026 |
| **Data Source** | DEFRA UK-AIR Portal |
| **AQI Scale** | UK DAQI (Low / Moderate / High / Very High) |

---

3  Project Overview

Air pollution is one of the most important environmental health
hazards in the United Kingdom. Public Health England (2019) estimated
that around 28,000 to 36,000 lives are lost annually in England due
to outdoor air pollution.

This project addresses the lack of publicly available, multi-city AQI
prediction systems by building a complete end-to-end machine learning
pipeline that:

- Collects and preprocesses historical hourly pollutant data
  (PM2.5, PM10, NO2, SO2, O3) from the official DEFRA UK-AIR portal
- Computes AQI categories aligned with the UK Daily Air Quality
  Index (DAQI) scale
- Engineers temporal, lag, and rolling mean features to capture
  pollution dynamics
- Trains and compares three classification models
- Deploys predictions as an interactive Streamlit web application

---

4  Cities Covered

| | | |
|---|---|---|
| 🏙️ London | 🏙️ Manchester | 🏙️ Birmingham |
| 🏙️ Leeds | 🏙️ Liverpool | 🏙️ Sheffield |
| 🏙️ Bristol | 🏙️ Glasgow | 🏙️ Edinburgh |
| 🏙️ Cardiff | 🏙️ Newcastle | |

---

5  Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **XGBoost** | **99.5%** | **99.4%** | **99.3%** | **99.3%** |
| Random Forest | 98.2% | 97.9% | 97.6% | 97.7% |
| LSTM | 95.8% | 95.1% | 94.9% | 95.0% |

6  Highlights
- ✅ XGBoost achieved **100% recall** on the Very High AQI class
- ✅ SMOTE oversampling applied to handle class imbalance (68% Low)
- ✅ 33 engineered features including lag variables up to 24 hours
- ✅ Macro-averaged metrics used for rigorous imbalanced evaluation
- ✅ Full pipeline from raw DEFRA data to deployed web application
- ✅ Interactive Streamlit app with real-time prediction and forecasting



7  Feature Engineering

| Feature Type | Description | Count |
|---|---|---|
| Temporal | Hour, day, month, season, day of week | 5 |
| Lag features | PM2.5, PM10, NO2 at 1,2,3,6,12,24 hrs | 18 |
| Rolling means | 6-hour and 24-hour windows for PM2.5, PM10, NO2 | 6 |
| Location | City label encoding | 1 |
| Raw pollutants | PM2.5, PM10, NO2 | 3 |
| **Total** | | **33** |

---

8  AQI Classification — UK DAQI Scale

| Category | DAQI Band | PM2.5 (µg/m³) | PM10 (µg/m³) | NO2 (µg/m³) | Health Advice |
|---|---|---|---|---|---|
| 🟢 Low | 1–3 | ≤ 11 | ≤ 16 | ≤ 67 | Enjoy usual outdoor activities |
| 🟡 Moderate | 4–6 | ≤ 23 | ≤ 33 | ≤ 134 | Reduce strenuous activity if symptomatic |
| 🟠 High | 7–9 | ≤ 35 | ≤ 50 | ≤ 200 | Reduce physical exertion outdoors |
| 🔴 Very High | 10 | > 35 | > 50 | > 200 | Avoid strenuous outdoor activity |

---

9  Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard
Process for Data Mining) framework:
