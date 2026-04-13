# UK Daily Air Quality Index (DAQI) Thresholds

## Overview

The DAQI is the UK standard for communicating air quality
to the public. Published by DEFRA and updated in line with
WHO guidelines (2022).

The DAQI assigns the overall index value based on the
maximum of the monitored pollutant sub-indices.

---

## Health Bands

| Band | Index | Colour | Health Advice |
|---|---|---|---|
| Low | 1–3 | 🟢 Green | Enjoy usual outdoor activities |
| Moderate | 4–6 | 🟡 Yellow | Reduce strenuous activity if experiencing symptoms |
| High | 7–9 | 🟠 Orange | Reduce physical exertion especially outdoors |
| Very High | 10 | 🔴 Red | Avoid strenuous outdoor activity |

---

## Full Classification Thresholds

### PM2.5 (Fine Particulate Matter) µg/m³

| Band | Index | Min | Max |
|---|---|---|---|
| Low | 1 | 0 | 11 |
| Low | 2 | 12 | 23 |
| Low | 3 | 24 | 35 |
| Moderate | 4 | 36 | 41 |
| Moderate | 5 | 42 | 47 |
| Moderate | 6 | 48 | 53 |
| High | 7 | 54 | 58 |
| High | 8 | 59 | 64 |
| High | 9 | 65 | 70 |
| Very High | 10 | 71 | + |

### PM10 (Coarse Particulate Matter) µg/m³

| Band | Index | Min | Max |
|---|---|---|---|
| Low | 1 | 0 | 16 |
| Low | 2 | 17 | 33 |
| Low | 3 | 34 | 50 |
| Moderate | 4 | 51 | 58 |
| Moderate | 5 | 59 | 66 |
| Moderate | 6 | 67 | 75 |
| High | 7 | 76 | 83 |
| High | 8 | 84 | 91 |
| High | 9 | 92 | 100 |
| Very High | 10 | 101 | + |

### NO2 (Nitrogen Dioxide) µg/m³

| Band | Index | Min | Max |
|---|---|---|---|
| Low | 1 | 0 | 67 |
| Low | 2 | 68 | 134 |
| Low | 3 | 135 | 200 |
| Moderate | 4 | 201 | 267 |
| Moderate | 5 | 268 | 334 |
| Moderate | 6 | 335 | 400 |
| High | 7 | 401 | 467 |
| High | 8 | 468 | 534 |
| High | 9 | 535 | 600 |
| Very High | 10 | 601 | + |

### O3 (Ozone) µg/m³

| Band | Index | Min | Max |
|---|---|---|---|
| Low | 1 | 0 | 33 |
| Low | 2 | 34 | 66 |
| Low | 3 | 67 | 100 |
| Moderate | 4 | 101 | 120 |
| Moderate | 5 | 121 | 140 |
| Moderate | 6 | 141 | 160 |
| High | 7 | 161 | 187 |
| High | 8 | 188 | 213 |
| High | 9 | 214 | 240 |
| Very High | 10 | 241 | + |

### SO2 (Sulphur Dioxide) µg/m³

| Band | Index | Min | Max |
|---|---|---|---|
| Low | 1 | 0 | 88 |
| Low | 2 | 89 | 177 |
| Low | 3 | 178 | 266 |
| Moderate | 4 | 267 | 354 |
| Moderate | 5 | 355 | 443 |
| Moderate | 6 | 444 | 532 |
| High | 7 | 533 | 710 |
| High | 8 | 711 | 887 |
| High | 9 | 888 | 1064 |
| Very High | 10 | 1065 | + |

---

## How This Project Uses DAQI

- AQI labels computed using PM2.5, PM10 and NO2 thresholds
- Priority rule applied: record assigned highest category
  triggered by any single pollutant
- Four simplified bands used for classification:
  Low, Moderate, High, Very High
- SO2 and O3 excluded from labelling due to data availability
  constraints

---

## Vulnerable Groups

| Group | Risk |
|---|---|
| Asthma sufferers | High sensitivity to PM2.5 and NO2 |
| Cardiovascular patients | Long-term PM2.5 exposure risk |
| Elderly | Reduced respiratory resilience |
| Children | Developing lung function vulnerability |
| Outdoor workers | Extended exposure duration |

---

## Source

DEFRA (2022) UK Daily Air Quality Index
👉 https://uk-air.defra.gov.uk

Committee on the Medical Effects of Air Pollutants (COMEAP)
👉 https://www.gov.uk/government/organisations/committee-on-the-medical-effects-of-air-pollutants
