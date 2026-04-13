# Methodology

## Framework

This project follows the **CRISP-DM** (Cross-Industry Standard
Process for Data Mining) framework selected over alternatives
such as KDD and SEMMA because it explicitly focuses on
deployment preparedness.

---

## Research Design

| | |
|---|---|
| **Approach** | Quantitative, data-driven |
| **Framework** | CRISP-DM |
| **Philosophy** | Positivist and inductive |
| **Models** | Random Forest, XGBoost, LSTM |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score |

---

## Data Collection

| | |
|---|---|
| **Source** | DEFRA UK-AIR Portal |
| **Network** | Automatic Urban and Rural Network (AURN) |
| **Period** | January 2021 – February 2026 |
| **Cities** | 11 major UK cities |
| **Pollutants** | PM2.5, PM10, NO2, SO2, O3 |
| **Raw Records** | ~450,000 hourly measurements |

---

## Data Preprocessing Steps

### Step 1 — Column Standardisation
- DEFRA column headers mapped to standardised names
- PM2_5, PM10, NO2, SO2, O3
- Status indicator columns removed

### Step 2 — Datetime Parsing
- Date and Time fields combined into single Datetime field
- Rows with invalid timestamps removed
- Index confirmed monotonically ordered per city

### Step 3 — Missing Value Treatment
- Forward fill and backward fill for gaps under 48 hours
- Long gaps exceeding 48 hours imputed with zero values
- Only 2.3% of total rows affected by long gap imputation

### Step 4 — Multi-City Expansion
- Base dataset replicated into city-specific copies
- Independent Gaussian noise added per city
- Standard deviations: PM2.5 = 2, PM10 = 3, NO2 = 5 µg/m³
- Fixed random seed (42) used for reproducibility

### Step 5 — Outlier Detection
- Values above 99.9th percentile identified
- Retained rather than removed as genuine atmospheric events
- Examples: Saharan dust intrusions, bonfire seasons

---

## AQI Label Computation

Labels computed using UK DAQI concentration thresholds:

| Category | DAQI Band | PM2.5 (µg/m³) | PM10 (µg/m³) | NO2 (µg/m³) |
|---|---|---|---|---|
| Low | 1–3 | ≤ 11 | ≤ 16 | ≤ 67 |
| Moderate | 4–6 | ≤ 23 | ≤ 33 | ≤ 134 |
| High | 7–9 | ≤ 35 | ≤ 50 | ≤ 200 |
| Very High | 10 | > 35 | > 50 | > 200 |

**Priority rule:** Record placed in highest category
triggered by any single pollutant.

**Label distribution after computation:**

| Category | Proportion |
|---|---|
| Low | ~68% |
| Moderate | ~21% |
| High | ~9% |
| Very High | ~2% |

---

## Feature Engineering

### Temporal Features (5)
| Feature | Description |
|---|---|
| Hour | Hour of day (0–23) |
| Day | Day of month (1–31) |
| Month | Month of year (1–12) |
| Day of week | Monday=0, Sunday=6 |
| Season | Winter, Spring, Summer, Autumn |

### Lag Features (18)
PM2.5, PM10, NO2 values at intervals of:
1, 2, 3, 6, 12, and 24 hours

### Rolling Mean Features (6)
6-hour and 24-hour rolling means for PM2.5, PM10, NO2

### Location Encoding (1)
City names ordinally encoded using scikit-learn LabelEncoder

### Total: 33 features

---

## Training and Evaluation Protocol

| | |
|---|---|
| **Split** | 80% training / 20% validation |
| **Method** | Stratified random split |
| **Class balancing** | SMOTE on training set only |
| **Test set** | All 2026 data held out |

### Evaluation Metrics
| Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified instances |
| Precision | Positive predictive value (macro-averaged) |
| Recall | Sensitivity per class (macro-averaged) |
| F1-Score | Harmonic mean of precision and recall |

---

## Limitations

- Synthetic multi-city expansion does not reflect genuine
  city-specific pollution characteristics
- Meteorological variables not included
- LSTM uses single-timestep input configuration
- Validation on SMOTE-balanced distribution inflates
  reported accuracy vs real-world deployment
- SO2 and O3 excluded from feature set
- Stratified random split rather than temporal split
  used for validation
