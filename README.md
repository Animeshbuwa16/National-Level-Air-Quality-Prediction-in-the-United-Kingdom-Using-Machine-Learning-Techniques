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

9 Installation & Usage

### Prerequisites
- Python 3.13
- pip

**Clone the Repository
```bash
git clone https://github.com/yourusername/uk-aqi-prediction.git
cd uk-aqi-prediction
```

** Install Dependencies
```bash
pip install -r requirements.txt
```
 Step 1 — Download Data
Download hourly pollutant data from the DEFRA UK-AIR portal:
👉 https://uk-air.defra.gov.uk/data/data-selector

Place the CSV file in the `/data` folder.
See `data/README.md` for detailed instructions.

 Step 2 — Train the Models
```bash
python src/train_model.py
```

 Step 3 — Launch the App
```bash
streamlit run src/app.py
```

---

10 Technology Stack

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.x | Data manipulation and preprocessing |
| NumPy | 1.26.x | Numerical operations |
| scikit-learn | 1.4.x | Random Forest, preprocessing, metrics |
| XGBoost | 2.x | Gradient boosting classifier |
| TensorFlow/Keras | 2.16.x | LSTM deep learning model |
| imbalanced-learn | 0.12.x | SMOTE class balancing |
| Streamlit | 1.35.x | Web application deployment |
| Matplotlib/Seaborn | 3.8.x | Visualisation |
| joblib | 1.3.x | Model serialisation and caching |

---

11 Limitations & Future Work

### Current Limitations
- Multi-city data uses synthetic Gaussian expansion due to limited
  AURN coverage outside London and Manchester
- Meteorological variables (wind speed, temperature, humidity,
  pressure) not included in current feature set
- LSTM uses single-timestep input; full sequence modelling would
  improve performance
- Validation metrics reflect SMOTE-balanced distribution rather
  than natural class proportions

12 Recommended Future Work
- Replace synthetic expansion with genuine city-specific AURN data
- Integrate Met Office open data API for meteorological features
- Retrain LSTM with sliding window multi-timestep sequence input
- Apply conformal prediction for calibrated uncertainty intervals
- Connect to DEFRA real-time data feed for live predictions
- Conduct equity audit across deprived and affluent monitoring areas

---

13 Data Source & Licence

All data sourced from the **DEFRA UK-AIR Portal**
- https://uk-air.defra.gov.uk

Published under the **Open Government Licence v3.0**
- https://www.nationalarchives.gov.uk/doc/open-government-licence

Raw data files are not included in this repository due to file size.

---
 14 Academic Context

This project was developed in partial fulfilment of the requirements
for the degree of **MSc Data Science** at Coventry University.


| **Module** | 7150CEM Data Science Project |
| **Submission** | April 2026 |
| **Ethics Approval** | P193103 |
| **Supervisor** | Omid Chatrabgoun |

---
15 Acknowledgements

- **DEFRA** and the UK-AIR monitoring network for open access data
- **Coventry University** School of Computing for project supervision
- Open source contributors of scikit-learn, XGBoost and Streamlit
- Public Health England for air quality health impact statistics

---

16 Contact

Animesh Buwa
MSc Data Science — Coventry University
Student ID: 15863968
---


