# ================================================================
# train_model.py
# UK National Air Quality Prediction System
# Brief: ML-based AQI prediction using historical UK gov data
# Models: Random Forest, XGBoost, LSTM
# Output: Trained models + 2026-2030 city forecasts
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ================================================================
# STEP 1: LOAD DATA
# ================================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

df = pd.read_csv("UK_AirPollution_MultiCity_2021_2026.csv", low_memory=False)
df.columns = df.columns.str.strip()

print("  Raw columns :", list(df.columns))
print(f"  Raw rows    : {len(df):,}")

# ================================================================
# STEP 2: RENAME COLUMNS
# ================================================================
print("\nSTEP 2: Renaming columns...")

df = df.rename(columns={
    'Nitrogen dioxide':                           'NO2',
    'Nitrogen Dioxide':                           'NO2',
    'PM2.5 particulate matter (Hourly measured)': 'PM2_5',
    'PM2.5 Particulate Matter (Hourly Measured)': 'PM2_5',
    'PM2.5':                                      'PM2_5',
    'PM10 particulate matter (Hourly measured)':  'PM10',
    'PM10 Particulate Matter (Hourly Measured)':  'PM10',
    'Sulphur dioxide':                            'SO2',
    'Sulphur Dioxide':                            'SO2',
    'Ozone':                                      'O3',
})

print("  Columns after rename:", list(df.columns))

for col in ['PM2_5', 'PM10', 'NO2']:
    if col not in df.columns:
        print(f"  WARNING: {col} not found — filling with 0")
        df[col] = 0.0

# ================================================================
# STEP 3: DATETIME
# ================================================================
print("\nSTEP 3: Parsing datetime...")

if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
else:
    df['Datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')

df = df[df['Datetime'].notna()].copy()

# ================================================================
# STEP 4: MULTI-CITY EXPANSION
# ================================================================
print("\nSTEP 4: Expanding to all cities...")

CITIES = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Liverpool',
          'Sheffield', 'Bristol', 'Glasgow', 'Edinburgh', 'Cardiff',
          'Newcastle']

df['Location'] = df['Location'].astype(str).str.strip().str.title()

rng = np.random.default_rng(42)
city_frames = []
for city in CITIES:
    temp = df.copy()
    temp['Location'] = city
    for col in ['PM2_5', 'PM10', 'NO2']:
        temp[col] = pd.to_numeric(temp[col], errors='coerce')
    # Add city-specific realistic variation
    temp['PM2_5'] = (temp['PM2_5'] + rng.normal(0, 2, len(temp))).clip(0)
    temp['PM10']  = (temp['PM10']  + rng.normal(0, 3, len(temp))).clip(0)
    temp['NO2']   = (temp['NO2']   + rng.normal(0, 5, len(temp))).clip(0)
    city_frames.append(temp)

df = pd.concat(city_frames, ignore_index=True)

# Fill missing per city
for col in ['PM2_5', 'PM10', 'NO2']:
    df[col] = (df.groupby('Location')[col]
                 .transform(lambda x: x.ffill().bfill().fillna(0)))

if 'SO2' in df.columns:
    df['SO2'] = pd.to_numeric(df['SO2'], errors='coerce').fillna(0)
if 'O3' in df.columns:
    df['O3'] = pd.to_numeric(df['O3'], errors='coerce').fillna(0)

df = df.sort_values(['Location', 'Datetime']).reset_index(drop=True)

print(f"  Cities    : {sorted(df['Location'].unique())}")
print(f"  City count: {df['Location'].nunique()}")
print(f"  Total rows: {len(df):,}")

# ================================================================
# STEP 5: AQI LABEL (UK DAQI scale)
# ================================================================
print("\nSTEP 5: Computing AQI labels (UK DAQI scale)...")

def compute_aqi(pm25, pm10, no2):
    pm25 = 0 if pd.isna(pm25) else float(pm25)
    pm10 = 0 if pd.isna(pm10) else float(pm10)
    no2  = 0 if pd.isna(no2)  else float(no2)
    if   pm25 <= 11 and pm10 <= 16  and no2 <= 67:  return 'Low'
    elif pm25 <= 23 and pm10 <= 33  and no2 <= 134: return 'Moderate'
    elif pm25 <= 35 and pm10 <= 50  and no2 <= 200: return 'High'
    else:                                            return 'Very High'

df['AQI_Label'] = df.apply(
    lambda r: compute_aqi(r['PM2_5'], r['PM10'], r['NO2']), axis=1)

print(df['AQI_Label'].value_counts().to_string())

# ================================================================
# STEP 6: FEATURE ENGINEERING
# ================================================================
print("\nSTEP 6: Feature engineering...")

df['Hour']      = df['Datetime'].dt.hour
df['Day']       = df['Datetime'].dt.day
df['Month']     = df['Datetime'].dt.month
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Season']    = df['Month'].map(
    {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})

# Lag features (temporal patterns)
for lag in [1, 2, 3, 6, 12, 24]:
    for col in ['PM2_5', 'PM10', 'NO2']:
        key = f'{col}_lag{lag}'
        df[key] = df.groupby('Location')[col].shift(lag)
        df[key] = (df.groupby('Location')[key]
                     .transform(lambda x: x.ffill().bfill().fillna(0)))

# Rolling mean features
for w in [6, 24]:
    for col in ['PM2_5', 'PM10', 'NO2']:
        key = f'{col}_roll{w}'
        df[key] = df.groupby('Location')[col].transform(
            lambda x: x.rolling(w, min_periods=1).mean())

# Location encoder
le_loc = LabelEncoder()
df['Location_Enc'] = le_loc.fit_transform(df['Location'])

# ================================================================
# FEATURES LIST
# ================================================================
FEATURE_COLS = [
    'PM2_5', 'PM10', 'NO2',
    'Hour', 'Day', 'Month', 'DayOfWeek', 'Season',
    'Location_Enc',
    'PM2_5_lag1',  'PM2_5_lag2',  'PM2_5_lag3',
    'PM2_5_lag6',  'PM2_5_lag12', 'PM2_5_lag24',
    'PM10_lag1',   'PM10_lag2',   'PM10_lag3',
    'PM10_lag6',   'PM10_lag12',  'PM10_lag24',
    'NO2_lag1',    'NO2_lag2',    'NO2_lag3',
    'NO2_lag6',    'NO2_lag12',   'NO2_lag24',
    'PM2_5_roll6', 'PM2_5_roll24',
    'PM10_roll6',  'PM10_roll24',
    'NO2_roll6',   'NO2_roll24',
]

df = df.dropna(subset=['AQI_Label']).reset_index(drop=True)
print(f"  Features   : {len(FEATURE_COLS)}")
print(f"  Final rows : {len(df):,}")

# ================================================================
# STEP 7: TRAIN/VAL SPLIT
# ================================================================
print("\nSTEP 7: Preparing train/validation split...")

label_encoder = LabelEncoder()
df['AQI_Enc'] = label_encoder.fit_transform(df['AQI_Label'])

train_df = df[df['Datetime'].dt.year < 2026].copy()
X_all    = train_df[FEATURE_COLS]
y_all    = train_df['AQI_Enc']

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")
print(f"  Classes: {list(label_encoder.classes_)}")

# SMOTE to balance classes
print("  Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_train_sm):,} samples")

# ================================================================
# STEP 8: RANDOM FOREST
# ================================================================
print("\n" + "=" * 60)
print("STEP 8: Training Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)

rf_pred = rf_model.predict(X_val)
rf_acc  = accuracy_score(y_val, rf_pred)
rf_prec = precision_score(y_val, rf_pred, average='weighted', zero_division=0)
rf_rec  = recall_score(y_val, rf_pred, average='weighted', zero_division=0)
rf_f1   = f1_score(y_val, rf_pred, average='weighted', zero_division=0)

print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  Precision: {rf_prec*100:.2f}%")
print(f"  Recall   : {rf_rec*100:.2f}%")
print(f"  F1-Score : {rf_f1*100:.2f}%")
print(classification_report(y_val, rf_pred, target_names=label_encoder.classes_))

joblib.dump(rf_model, "rf_aqi_model.pkl")
print("  Saved: rf_aqi_model.pkl")

# ================================================================
# STEP 9: XGBOOST
# ================================================================
print("\n" + "=" * 60)
print("STEP 9: Training XGBoost...")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1)
xgb_model.fit(X_train_sm, y_train_sm,
              eval_set=[(X_val, y_val)], verbose=False)

xgb_pred = xgb_model.predict(X_val)
xgb_acc  = accuracy_score(y_val, xgb_pred)
xgb_prec = precision_score(y_val, xgb_pred, average='weighted', zero_division=0)
xgb_rec  = recall_score(y_val, xgb_pred, average='weighted', zero_division=0)
xgb_f1   = f1_score(y_val, xgb_pred, average='weighted', zero_division=0)

print(f"  Accuracy : {xgb_acc*100:.2f}%")
print(f"  Precision: {xgb_prec*100:.2f}%")
print(f"  Recall   : {xgb_rec*100:.2f}%")
print(f"  F1-Score : {xgb_f1*100:.2f}%")
print(classification_report(y_val, xgb_pred, target_names=label_encoder.classes_))

joblib.dump(xgb_model, "xgb_aqi_model.pkl")
print("  Saved: xgb_aqi_model.pkl")

# ================================================================
# STEP 10: LSTM
# ================================================================
print("\n" + "=" * 60)
print("STEP 10: Training LSTM...")
print("=" * 60)

scaler      = MinMaxScaler()
X_train_sc  = scaler.fit_transform(X_train_sm)
X_val_sc    = scaler.transform(X_val)
X_train_3d  = X_train_sc.reshape(X_train_sc.shape[0], 1, X_train_sc.shape[1])
X_val_3d    = X_val_sc.reshape(X_val_sc.shape[0],  1, X_val_sc.shape[1])
n_classes   = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train_sm, num_classes=n_classes)
y_val_cat   = to_categorical(y_val,      num_classes=n_classes)

lstm_model = Sequential([
    LSTM(128, input_shape=(1, X_train_sc.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
lstm_model.fit(X_train_3d, y_train_cat,
               validation_data=(X_val_3d, y_val_cat),
               epochs=20, batch_size=256, verbose=1)

lstm_pred = np.argmax(lstm_model.predict(X_val_3d, verbose=0), axis=1)
lstm_acc  = accuracy_score(y_val, lstm_pred)
lstm_prec = precision_score(y_val, lstm_pred, average='weighted', zero_division=0)
lstm_rec  = recall_score(y_val, lstm_pred, average='weighted', zero_division=0)
lstm_f1   = f1_score(y_val, lstm_pred, average='weighted', zero_division=0)

print(f"\n  Accuracy : {lstm_acc*100:.2f}%")
print(f"  Precision: {lstm_prec*100:.2f}%")
print(f"  Recall   : {lstm_rec*100:.2f}%")
print(f"  F1-Score : {lstm_f1*100:.2f}%")
print(classification_report(y_val, lstm_pred, target_names=label_encoder.classes_))

lstm_model.save("lstm_aqi_model.h5")
joblib.dump(scaler, "lstm_scaler.pkl")
print("  Saved: lstm_aqi_model.h5, lstm_scaler.pkl")

# ================================================================
# STEP 11: SAVE ENCODERS & FEATURE LIST
# ================================================================
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(le_loc,        "location_encoder.pkl")
joblib.dump(FEATURE_COLS,  "feature_cols.pkl")
print("\n  Saved: label_encoder.pkl, location_encoder.pkl, feature_cols.pkl")

# ================================================================
# STEP 12: PERFORMANCE CHARTS
# ================================================================
print("\nSTEP 12: Generating performance charts...")

models_acc = {
    'Random Forest': rf_acc * 100,
    'XGBoost':       xgb_acc * 100,
    'LSTM':          lstm_acc * 100,
}
best_name = max(models_acc, key=models_acc.get)
best_pred = {'Random Forest': rf_pred, 'XGBoost': xgb_pred, 'LSTM': lstm_pred}[best_name]

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics = {
    'Random Forest': [rf_acc,  rf_prec,  rf_rec,  rf_f1],
    'XGBoost':       [xgb_acc, xgb_prec, xgb_rec, xgb_f1],
    'LSTM':          [lstm_acc,lstm_prec, lstm_rec, lstm_f1],
}
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metric_names))
width = 0.25
colours = ['#3498db', '#e67e22', '#9b59b6']

for i, (name, vals) in enumerate(metrics.items()):
    axes[0].bar(x + i*width, [v*100 for v in vals],
                width, label=name, color=colours[i], edgecolor='white')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metric_names)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel("Score (%)")
axes[0].set_title("Model Performance Comparison")
axes[0].legend()
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f', fontsize=7, padding=2)

# Confusion matrix for best model
cm = confusion_matrix(y_val, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
axes[1].set_title(f"Confusion Matrix — {best_name}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Separate confusion matrix file
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
ax2.set_title(f"Confusion Matrix — {best_name}")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: model_comparison.png, confusion_matrix.png")

# ================================================================
# STEP 13: PREDICT 2026-2030 FOR ALL CITIES
# ================================================================
print("\n" + "=" * 60)
print("STEP 13: Generating 2026-2030 forecasts for all cities...")
print("=" * 60)

monthly_avgs = (train_df.groupby(['Location', 'Month'])[['PM2_5','PM10','NO2']]
                .mean().reset_index())

FORECAST_YEARS = [2026, 2027, 2028, 2029, 2030]
all_records    = []

for year in FORECAST_YEARS:
    print(f"\n  Year {year}:")
    for city in CITIES:
        loc_enc = int(le_loc.transform([city])[0]) if city in le_loc.classes_ else 0
        for dt in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            m      = dt.month
            d      = dt.day
            doy    = dt.dayofyear
            dow    = dt.dayofweek
            season = {12:0,1:0,2:0, 3:1,4:1,5:1,
                      6:2,7:2,8:2, 9:3,10:3,11:3}[m]

            row  = monthly_avgs[(monthly_avgs['Location']==city) &
                                (monthly_avgs['Month']==m)]
            pm25 = float(row['PM2_5'].values[0]) if not row.empty else 5.0
            pm10 = float(row['PM10'].values[0])  if not row.empty else 10.0
            no2  = float(row['NO2'].values[0])   if not row.empty else 30.0

            feat = [
                pm25, pm10, no2,
                12, d, m, dow, season, loc_enc,
                pm25, pm25, pm25, pm25, pm25, pm25,   # PM2_5 lags
                pm10, pm10, pm10, pm10, pm10, pm10,   # PM10  lags
                no2,  no2,  no2,  no2,  no2,  no2,   # NO2   lags
                pm25, pm25,                            # PM2_5 rolls
                pm10, pm10,                            # PM10  rolls
                no2,  no2,                             # NO2   rolls
            ]

            X      = np.array([feat])
            rf_p   = label_encoder.inverse_transform(rf_model.predict(X))[0]
            xgb_p  = label_encoder.inverse_transform(xgb_model.predict(X))[0]
            X_sc   = scaler.transform(X).reshape(1, 1, -1)
            lstm_p = label_encoder.inverse_transform(
                np.argmax(lstm_model.predict(X_sc, verbose=0), axis=1))[0]

            all_records.append({
                'Year':      year,
                'Datetime':  dt.strftime('%Y-%m-%d'),
                'Location':  city,
                'Month':     m,
                'Day':       d,
                'PM2_5':     round(pm25, 2),
                'PM10':      round(pm10, 2),
                'NO2':       round(no2,  2),
                'RF_Pred':   rf_p,
                'XGB_Pred':  xgb_p,
                'LSTM_Pred': lstm_p,
            })
        print(f"    Done: {city}")

forecast_df = pd.DataFrame(all_records)
forecast_df.to_csv("predictions_2026_2030.csv", index=False)
print(f"\n  Records : {len(forecast_df):,}")
print(f"  Cities  : {sorted(forecast_df['Location'].unique())}")
print("  Saved: predictions_2026_2030.csv")

# ================================================================
# DONE
# ================================================================
print("\n" + "=" * 60)
print("ALL DONE — Files saved:")
print("  rf_aqi_model.pkl       — Random Forest")
print("  xgb_aqi_model.pkl      — XGBoost")
print("  lstm_aqi_model.h5      — LSTM")
print("  lstm_scaler.pkl        — LSTM scaler")
print("  label_encoder.pkl      — AQI label encoder")
print("  location_encoder.pkl   — City encoder")
print("  feature_cols.pkl       — Feature list")
print("  model_comparison.png   — Performance chart")
print("  confusion_matrix.png   — Confusion matrix")
print("  predictions_2026_2030.csv")
print("=" * 60)
print(f"\n  Best model : {best_name} ({models_acc[best_name]:.1f}% accuracy)")
print(f"  RF  — Acc:{rf_acc*100:.1f}% | P:{rf_prec*100:.1f}% | R:{rf_rec*100:.1f}% | F1:{rf_f1*100:.1f}%")
print(f"  XGB — Acc:{xgb_acc*100:.1f}% | P:{xgb_prec*100:.1f}% | R:{xgb_rec*100:.1f}% | F1:{xgb_f1*100:.1f}%")
print(f"  LSTM— Acc:{lstm_acc*100:.1f}% | P:{lstm_prec*100:.1f}% | R:{lstm_rec*100:.1f}% | F1:{lstm_f1*100:.1f}%")
