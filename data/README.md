# Data

## Source
All data used in this project was sourced from the official
DEFRA UK-AIR portal:
👉 https://uk-air.defra.gov.uk/data/data-selector

## How to Download Data

1. Go to https://uk-air.defra.gov.uk/data/data-selector
2. Select the following cities:
   - London
   - Manchester
   - Birmingham
   - Leeds
   - Liverpool
   - Sheffield
   - Bristol
   - Glasgow
   - Edinburgh
   - Cardiff
   - Newcastle
3. Select pollutants:
   - PM2.5
   - PM10
   - NO2
   - SO2
   - O3
4. Select date range: **January 2021 to February 2026**
5. Download as CSV
6. Place the CSV file in this `/data` folder
7. Run `src/train_model.py`

## Dataset Details

| | |
|---|---|
| **Source** | DEFRA UK-AIR Portal |
| **Coverage** | 11 major UK cities |
| **Period** | January 2021 – February 2026 |
| **Frequency** | Hourly measurements |
| **Pollutants** | PM2.5, PM10, NO2, SO2, O3 |
| **Raw Records** | ~450,000 hourly rows |
| **Network** | Automatic Urban and Rural Network (AURN) |

## Pollutants Description

| Pollutant | Full Name | Health Impact |
|---|---|---|
| PM2.5 | Fine Particulate Matter | Respiratory and cardiovascular disease |
| PM10 | Coarse Particulate Matter | Lung inflammation |
| NO2 | Nitrogen Dioxide | Respiratory inflammation |
| SO2 | Sulphur Dioxide | Airway irritation |
| O3 | Ozone | Lung damage at high levels |

## Data Quality

- Status indicator columns are automatically removed during
  preprocessing
- Missing values treated with forward/backward fill for gaps
  under 48 hours
- Long gaps exceeding 48 hours imputed with zero values
- Outliers above 99.9th percentile retained as genuine
  atmospheric events

## Why Raw Data is Not Included

Raw data files are not included in this repository because:
- File size is too large (~450,000 rows)
- Data is freely available from DEFRA under Open Government
  Licence v3.0
- Redistribution is unnecessary given open access availability

## Licence

Data is published under the **Open Government Licence v3.0**
👉 https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

This licence permits use, adaptation and redistribution under
both non-commercial and commercial terms subject to appropriate
acknowledgement.# Data

## Source
All data used in this project was sourced from the official
DEFRA UK-AIR portal:
👉 https://uk-air.defra.gov.uk/data/data-selector

## How to Download Data

1. Go to https://uk-air.defra.gov.uk/data/data-selector
2. Select the following cities:
   - London
   - Manchester
   - Birmingham
   - Leeds
   - Liverpool
   - Sheffield
   - Bristol
   - Glasgow
   - Edinburgh
   - Cardiff
   - Newcastle
3. Select pollutants:
   - PM2.5
   - PM10
   - NO2
   - SO2
   - O3
4. Select date range: **January 2021 to February 2026**
5. Download as CSV
6. Place the CSV file in this `/data` folder
7. Run `src/train_model.py`

## Dataset Details

| | |
|---|---|
| **Source** | DEFRA UK-AIR Portal |
| **Coverage** | 11 major UK cities |
| **Period** | January 2021 – February 2026 |
| **Frequency** | Hourly measurements |
| **Pollutants** | PM2.5, PM10, NO2, SO2, O3 |
| **Raw Records** | ~450,000 hourly rows |
| **Network** | Automatic Urban and Rural Network (AURN) |

## Pollutants Description

| Pollutant | Full Name | Health Impact |
|---|---|---|
| PM2.5 | Fine Particulate Matter | Respiratory and cardiovascular disease |
| PM10 | Coarse Particulate Matter | Lung inflammation |
| NO2 | Nitrogen Dioxide | Respiratory inflammation |
| SO2 | Sulphur Dioxide | Airway irritation |
| O3 | Ozone | Lung damage at high levels |

## Data Quality

- Status indicator columns are automatically removed during
  preprocessing
- Missing values treated with forward/backward fill for gaps
  under 48 hours
- Long gaps exceeding 48 hours imputed with zero values
- Outliers above 99.9th percentile retained as genuine
  atmospheric events

## Why Raw Data is Not Included

Raw data files are not included in this repository because:
- File size is too large (~450,000 rows)
- Data is freely available from DEFRA under Open Government
  Licence v3.0
- Redistribution is unnecessary given open access availability

## Licence

Data is published under the **Open Government Licence v3.0**
👉 https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

This licence permits use, adaptation and redistribution under
both non-commercial and commercial terms subject to appropriate
acknowledgement.
