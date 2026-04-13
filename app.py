import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import joblib
import io

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="UK AQI Research Dashboard", layout="wide")
st.title("UK National Air Quality Research Dashboard")

AQI_COLOURS = {
    "Low":       "#2ecc71",
    "Moderate":  "#f39c12",
    "High":      "#e74c3c",
    "Very High": "#8e44ad",
    "Unknown":   "#aaaaaa",
}
AQI_ORDER   = ["Low", "Moderate", "High", "Very High"]
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

CITIES = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Liverpool',
          'Sheffield', 'Bristol', 'Glasgow', 'Edinburgh', 'Cardiff',
          'Newcastle']

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

# ================================================================
# DATA LOADING
# ================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("UK_AirPollution_MultiCity_2021_2026.csv", low_memory=False)
        df.columns = df.columns.str.strip()

        df = df.rename(columns={
            'Nitrogen dioxide':                           'NO2',
            'Nitrogen Dioxide':                           'NO2',
            'PM2.5 particulate matter (Hourly measured)': 'PM2_5',
            'PM2.5':                                      'PM2_5',
            'PM10 particulate matter (Hourly measured)':  'PM10',
            'Sulphur dioxide':                            'SO2',
            'Ozone':                                      'O3',
        })

        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            df['Datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce')
        df = df[df['Datetime'].notna()].copy()

        # Multi-city expansion (mirrors train_model.py)
        rng = np.random.default_rng(42)
        city_frames = []
        for city in CITIES:
            temp = df.copy()
            temp['Location'] = city
            for col in ['PM2_5', 'PM10', 'NO2']:
                temp[col] = pd.to_numeric(temp[col], errors='coerce')
            temp['PM2_5'] = (temp['PM2_5'] + rng.normal(0, 2, len(temp))).clip(0)
            temp['PM10']  = (temp['PM10']  + rng.normal(0, 3, len(temp))).clip(0)
            temp['NO2']   = (temp['NO2']   + rng.normal(0, 5, len(temp))).clip(0)
            city_frames.append(temp)

        df = pd.concat(city_frames, ignore_index=True)

        for col in ['PM2_5', 'PM10', 'NO2']:
            df[col] = (df.groupby('Location')[col]
                         .transform(lambda x: x.ffill().bfill().fillna(0)))

        df['Hour']      = df['Datetime'].dt.hour
        df['Day']       = df['Datetime'].dt.day
        df['Month']     = df['Datetime'].dt.month
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        df['Season']    = df['Month'].map(
            {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})

        for lag in [1, 2, 3, 6, 12, 24]:
            for col in ['PM2_5', 'PM10', 'NO2']:
                key = f'{col}_lag{lag}'
                df[key] = df.groupby('Location')[col].shift(lag)
                df[key] = (df.groupby('Location')[key]
                             .transform(lambda x: x.ffill().bfill().fillna(0)))

        for w in [6, 24]:
            for col in ['PM2_5', 'PM10', 'NO2']:
                key = f'{col}_roll{w}'
                df[key] = df.groupby('Location')[col].transform(
                    lambda x: x.rolling(w, min_periods=1).mean())

        try:
            le_loc  = joblib.load("location_encoder.pkl")
            loc_map = {cls: idx for idx, cls in enumerate(le_loc.classes_)}
        except Exception:
            loc_map = {c: i for i, c in enumerate(sorted(CITIES))}

        df['Location_Enc'] = df['Location'].map(loc_map).fillna(0).astype(int)
        df = df.sort_values(['Location','Datetime']).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_forecast():
    for fname in ["predictions_2026_2030.csv", "predictions_2026.csv"]:
        try:
            fc = pd.read_csv(fname)
            fc['Datetime'] = pd.to_datetime(fc['Datetime'])
            if 'Year' not in fc.columns:
                fc['Year'] = fc['Datetime'].dt.year
            fc['Location'] = fc['Location'].astype(str).str.strip().str.title()
            return fc
        except FileNotFoundError:
            continue
    return None


@st.cache_resource
def load_models():
    models        = {}
    label_encoder = None

    try:
        label_encoder = joblib.load("label_encoder.pkl")
    except Exception as e:
        st.error(f"label_encoder.pkl not found: {e}")

    for fname in ["rf_aqi_model.pkl", "rf_model.pkl"]:
        try:
            models['Random Forest'] = joblib.load(fname); break
        except FileNotFoundError:
            pass
    if 'Random Forest' not in models:
        models['Random Forest'] = None

    for fname in ["xgb_aqi_model.pkl", "xgb_model.pkl"]:
        try:
            models['XGBoost'] = joblib.load(fname); break
        except FileNotFoundError:
            pass
    if 'XGBoost' not in models:
        models['XGBoost'] = None

    try:
        from tensorflow.keras.models import load_model
        models['LSTM'] = load_model("lstm_aqi_model.h5")
    except Exception:
        models['LSTM'] = None

    return models, label_encoder


# ── Boot ─────────────────────────────────────────────────────
df       = load_data()
forecast = load_forecast()

if df.empty:
    st.error("No data loaded. Check CSV file.")
    st.stop()

models, label_encoder = load_models()
available_models = [k for k, v in models.items() if v is not None]

if not available_models:
    st.error("No models found. Run train_model.py first.")
    st.stop()

if label_encoder is None:
    st.error("label_encoder.pkl missing. Run train_model.py first.")
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("Dashboard Controls")

all_cities = sorted(df['Location'].unique())
st.sidebar.info(f"Available cities: {len(all_cities)}")

city         = st.sidebar.selectbox("Select City", all_cities)
min_date     = df['Datetime'].min().date()
max_date     = df['Datetime'].max().date()
default_date = min(pd.Timestamp("2024-06-01").date(), max_date)
selected_date = st.sidebar.date_input(
    "Select Date", value=default_date,
    min_value=min_date, max_value=pd.Timestamp("2030-12-31").date())
hour         = st.sidebar.slider("Hour", 0, 23, 12)
model_choice = st.sidebar.selectbox("Select Model", available_models)

st.sidebar.divider()
st.sidebar.markdown("### Models Status")
for name, m in models.items():
    st.sidebar.markdown(f"{'✅' if m is not None else '❌'} {name}")

st.sidebar.divider()
st.sidebar.markdown(f"**Cities:** {len(all_cities)}")
st.sidebar.markdown(f"**Data range:** {min_date} to {max_date}")
st.sidebar.markdown(f"**Total records:** {len(df):,}")

if forecast is not None:
    fc_years = sorted(forecast['Year'].unique())
    st.sidebar.success(f"Forecast loaded: {fc_years[0]}-{fc_years[-1]}")
else:
    st.sidebar.error("No forecast file. Run train_model.py")

# ================================================================
# CITY DATA
# ================================================================
city_data = df[df['Location'] == city].copy()
if city_data.empty:
    st.warning(f"No data for {city}")
    st.stop()

selected_dt = pd.Timestamp(str(selected_date)) + pd.Timedelta(hours=hour)
hist_row    = city_data[city_data['Datetime'] == selected_dt]

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "Live Dashboard",
    "2026-2030 Forecast",
    "Model Performance"
])

# ================================================================
# TAB 1 — LIVE DASHBOARD
# ================================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pollutant Readings")

        if not hist_row.empty:
            pm25_val = float(hist_row['PM2_5'].values[0])
            pm10_val = float(hist_row['PM10'].values[0])
            no2_val  = float(hist_row['NO2'].values[0])
            lag_vals = {f: float(hist_row[f].values[0])
                        for f in FEATURE_COLS if f in hist_row.columns}
            st.info(f"Exact record found for {selected_dt}")
        else:
            month_data = city_data[city_data['Datetime'].dt.month == selected_date.month]
            avg        = month_data[['PM2_5','PM10','NO2']].mean()
            pm25_val   = float(avg.get('PM2_5', 5.0) or 5.0)
            pm10_val   = float(avg.get('PM10',  10.0) or 10.0)
            no2_val    = float(avg.get('NO2',   30.0) or 30.0)
            lag_vals   = {}
            lbl = ("historical average" if selected_date.year <= max_date.year
                   else f"{selected_date.year} forecast")
            st.info(f"Using {lbl} for {city}")

        m1, m2, m3 = st.columns(3)
        m1.metric("PM2.5 (ug/m3)", f"{pm25_val:.1f}")
        m2.metric("PM10 (ug/m3)",  f"{pm10_val:.1f}")
        m3.metric("NO2 (ug/m3)",   f"{no2_val:.1f}")

        st.subheader("PM2.5 Trend (Last 300 Records)")
        recent = city_data[['Datetime','PM2_5']].dropna().tail(300)
        if not recent.empty:
            fig1, ax1 = plt.subplots(figsize=(7, 3))
            ax1.plot(recent['Datetime'], recent['PM2_5'],
                     color='steelblue', linewidth=1)
            ax1.fill_between(recent['Datetime'], recent['PM2_5'],
                             alpha=0.15, color='steelblue')
            ax1.set_title(f"Recent PM2.5 - {city}", fontsize=11)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("PM2.5 (ug/m3)")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig1.autofmt_xdate(rotation=30)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

    with col2:
        st.subheader(f"AQI Prediction - {model_choice}")

        try:
            le_loc  = joblib.load("location_encoder.pkl")
            loc_enc = int(le_loc.transform([city])[0]) if city in le_loc.classes_ else 0
        except Exception:
            loc_enc = CITIES.index(city) if city in CITIES else 0

        dow    = selected_dt.dayofweek
        season = {12:0,1:0,2:0, 3:1,4:1,5:1,
                  6:2,7:2,8:2, 9:3,10:3,11:3}[selected_date.month]

        # Build feature vector matching FEATURE_COLS exactly
        feat_map = {
            'PM2_5': pm25_val, 'PM10': pm10_val, 'NO2': no2_val,
            'Hour': hour, 'Day': selected_date.day,
            'Month': selected_date.month, 'DayOfWeek': dow,
            'Season': season, 'Location_Enc': loc_enc,
        }
        # Fill lag/roll features
        for lag in [1,2,3,6,12,24]:
            feat_map[f'PM2_5_lag{lag}'] = lag_vals.get(f'PM2_5_lag{lag}', pm25_val)
            feat_map[f'PM10_lag{lag}']  = lag_vals.get(f'PM10_lag{lag}',  pm10_val)
            feat_map[f'NO2_lag{lag}']   = lag_vals.get(f'NO2_lag{lag}',   no2_val)
        for w in [6, 24]:
            feat_map[f'PM2_5_roll{w}'] = lag_vals.get(f'PM2_5_roll{w}', pm25_val)
            feat_map[f'PM10_roll{w}']  = lag_vals.get(f'PM10_roll{w}',  pm10_val)
            feat_map[f'NO2_roll{w}']   = lag_vals.get(f'NO2_roll{w}',   no2_val)

        input_vec  = np.array([[feat_map[f] for f in FEATURE_COLS]])
        sel_model  = models[model_choice]
        pred_label = "Unknown"
        confidence = 0.0
        probs      = None

        try:
            if model_choice == 'LSTM':
                scaler     = joblib.load("lstm_scaler.pkl")
                X_sc       = scaler.transform(input_vec).reshape(1, 1, -1)
                raw_probs  = sel_model.predict(X_sc, verbose=0)
                pred_idx   = int(np.argmax(raw_probs))
                confidence = float(np.max(raw_probs)) * 100
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                probs      = raw_probs[0]
            else:
                pred_enc   = sel_model.predict(input_vec).astype(int)
                pred_label = label_encoder.inverse_transform(pred_enc)[0]
                raw_probs  = sel_model.predict_proba(input_vec)
                confidence = float(np.max(raw_probs)) * 100
                probs      = raw_probs[0]

            box_col = AQI_COLOURS.get(pred_label, "#3498db")
            st.markdown(f"""
            <div style="background:{box_col};padding:24px;border-radius:12px;
                        text-align:center;margin-bottom:14px;">
              <h2 style="color:white;margin:0;font-size:2rem;">AQI: {pred_label}</h2>
              <p style="color:white;font-size:16px;margin:6px 0 0;">
                Confidence: {confidence:.1f}% &nbsp;|&nbsp; {model_choice}
              </p>
            </div>""", unsafe_allow_html=True)

            if probs is not None:
                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame({
                    "AQI Class":   label_encoder.classes_,
                    "Probability": probs
                }).sort_values("Probability", ascending=True)
                fig_p, ax_p = plt.subplots(figsize=(6, 3))
                ax_p.barh(prob_df["AQI Class"], prob_df["Probability"],
                          color=[AQI_COLOURS.get(c,"#aaa") for c in prob_df["AQI Class"]])
                ax_p.set_xlim(0, 1)
                ax_p.set_xlabel("Probability")
                ax_p.set_title("Prediction Probabilities")
                plt.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Row 2
    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Yearly Average PM2.5")
        city_data['Year'] = city_data['Datetime'].dt.year
        yearly_avg = city_data.groupby('Year')['PM2_5'].mean().dropna()
        if not yearly_avg.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            bars = ax2.bar(yearly_avg.index.astype(str), yearly_avg.values,
                           color='cornflowerblue', edgecolor='white')
            ax2.set_ylim(0, yearly_avg.max() * 1.3)
            ax2.set_title(f"Yearly Avg PM2.5 - {city}")
            ax2.set_xlabel("Year"); ax2.set_ylabel("ug/m3")
            for bar, v in zip(bars, yearly_avg.values):
                ax2.text(bar.get_x()+bar.get_width()/2, v+yearly_avg.max()*0.02,
                         f"{v:.1f}", ha='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    with col4:
        st.subheader("Pollutant Correlation Heatmap")
        try:
            cols = [c for c in ['PM2_5','PM10','NO2','SO2','O3']
                    if c in city_data.columns]
            corr = city_data[cols].apply(pd.to_numeric, errors='coerce').corr()
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax4, linewidths=0.5, square=True)
            ax4.set_title("Pollutant Correlations")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        except Exception as e:
            st.error(f"Heatmap error: {e}")

    # Row 3
    st.divider()
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Feature Importance")
        if model_choice in ['Random Forest','XGBoost'] and sel_model is not None:
            try:
                importances = sel_model.feature_importances_
                top_idx     = np.argsort(importances)[-15:]
                top_feats   = [FEATURE_COLS[i] for i in top_idx]
                top_vals    = importances[top_idx]
                colours     = ['#e74c3c' if v == max(importances) else '#3498db'
                               for v in top_vals]
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                ax3.barh(top_feats, top_vals, color=colours)
                ax3.set_title(f"Top 15 Features - {model_choice}")
                ax3.set_xlabel("Importance")
                ax3.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
                st.caption("Red = most important feature")
            except Exception as e:
                st.error(f"Feature importance error: {e}")
        else:
            st.info("Feature importance available for RF and XGBoost only.")

    with col6:
        st.subheader("Model Confidence Comparison")
        if len(available_models) > 1:
            try:
                comp = {}
                for name in available_models:
                    m = models[name]
                    if m is None: continue
                    if name == 'LSTM':
                        sc = joblib.load("lstm_scaler.pkl")
                        p  = m.predict(sc.transform(input_vec).reshape(1,1,-1), verbose=0)
                        comp[name] = float(np.max(p)) * 100
                    else:
                        comp[name] = float(np.max(m.predict_proba(input_vec))) * 100

                fig5, ax5 = plt.subplots(figsize=(5, 3))
                colour_map = {'Random Forest':'#3498db','XGBoost':'#e67e22','LSTM':'#9b59b6'}
                bars = ax5.bar(list(comp.keys()), list(comp.values()),
                               color=[colour_map.get(k,'#aaa') for k in comp.keys()])
                ax5.set_ylim(0, 110)
                ax5.set_ylabel("Confidence (%)")
                ax5.set_title("Model Confidence Comparison")
                for bar, val in zip(bars, comp.values()):
                    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                             f"{val:.1f}%", ha='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close(fig5)
            except Exception as e:
                st.error(f"Comparison error: {e}")
        else:
            st.info("Load multiple models to see comparison.")


# ================================================================
# TAB 2 — FORECAST
# ================================================================
with tab2:
    st.header("AQI Forecast 2026-2030")
    st.markdown("Machine learning predictions for all UK cities based on 6 years of historical pollution data.")

    if forecast is None:
        st.error("No forecast file found. Run train_model.py first.")
        st.stop()

    pred_col_map = {
        'Random Forest': 'RF_Pred',
        'XGBoost':       'XGB_Pred',
        'LSTM':          'LSTM_Pred'
    }
    avail_pred = [k for k in pred_col_map if pred_col_map[k] in forecast.columns]
    if not avail_pred:
        st.error("No prediction columns found in forecast CSV.")
        st.stop()

    fc1, fc2, fc3 = st.columns(3)
    fc_cities = sorted(forecast['Location'].unique())
    fc_city   = fc1.selectbox("City",  fc_cities, key="fc_city")
    fc_model  = fc2.selectbox("Model", avail_pred, key="fc_model")
    fc_year   = fc3.selectbox("Year",  sorted(forecast['Year'].unique()), key="fc_year")
    pred_col  = pred_col_map[fc_model]

    city_fc = forecast[
        (forecast['Location']==fc_city) & (forecast['Year']==fc_year)].copy()
    if city_fc.empty:
        st.warning(f"No forecast data for {fc_city} in {fc_year}.")
        st.stop()

    city_fc['AQI']   = city_fc[pred_col]
    city_fc['Month'] = pd.to_datetime(city_fc['Datetime']).dt.month
    city_fc['Day']   = pd.to_datetime(city_fc['Datetime']).dt.day

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Days Forecast",    f"{len(city_fc):,}")
    k2.metric("Most Common AQI",  city_fc['AQI'].value_counts().idxmax())
    k3.metric("Low Days",         f"{(city_fc['AQI']=='Low').sum()}")
    k4.metric("High / Very High", f"{((city_fc['AQI']=='High')|(city_fc['AQI']=='Very High')).sum()}")

    st.divider()

    # Monthly distribution
    st.subheader(f"Monthly AQI Distribution - {fc_city} {fc_year}")
    monthly_counts = city_fc.groupby(['Month','AQI']).size().unstack(fill_value=0)
    order          = [c for c in AQI_ORDER if c in monthly_counts.columns]
    monthly_counts = monthly_counts.reindex(columns=order, fill_value=0)
    monthly_counts.index = [MONTH_NAMES[i-1] for i in monthly_counts.index]
    fig6, ax6 = plt.subplots(figsize=(13, 4))
    bottom = np.zeros(len(monthly_counts))
    for cls in order:
        ax6.bar(monthly_counts.index, monthly_counts[cls], bottom=bottom,
                label=cls, color=AQI_COLOURS.get(cls,"#aaa"))
        bottom += monthly_counts[cls].values
    ax6.set_title(f"Monthly AQI - {fc_city} {fc_year} ({fc_model})")
    ax6.set_xlabel("Month"); ax6.set_ylabel("Days")
    ax6.legend(loc='upper right', fontsize=9)
    plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

    # Calendar heatmap
    st.subheader(f"Daily AQI Calendar - {fc_city} {fc_year}")
    aqi_num_map = {k: i for i, k in enumerate(AQI_ORDER+["Unknown"])}
    city_fc['AQI_Num'] = city_fc['AQI'].map(aqi_num_map).fillna(4)
    matrix = np.full((12, 31), np.nan)
    for _, row in city_fc.iterrows():
        matrix[int(row['Month'])-1, int(row['Day'])-1] = row['AQI_Num']
    cmap  = ListedColormap([AQI_COLOURS.get(k,"#aaa") for k in AQI_ORDER+["Unknown"]])
    fig7, ax7 = plt.subplots(figsize=(16, 5))
    ax7.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=4)
    ax7.set_yticks(range(12)); ax7.set_yticklabels(MONTH_NAMES, fontsize=10)
    ax7.set_xticks(range(31)); ax7.set_xticklabels(range(1,32), fontsize=8)
    ax7.set_title(f"Daily AQI Calendar - {fc_city} {fc_year} ({fc_model})")
    ax7.set_xlabel("Day of Month")
    ax7.legend(handles=[mpatches.Patch(color=AQI_COLOURS.get(k,"#aaa"), label=k)
                        for k in AQI_ORDER], bbox_to_anchor=(1.01,1), fontsize=9)
    plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)

    st.divider()

    # Year-over-year
    st.subheader(f"Year-over-Year AQI Trend - {fc_city} (2026-2030)")
    city_all        = forecast[forecast['Location']==fc_city].copy()
    city_all['AQI'] = city_all[pred_col]
    yoy = city_all.groupby(['Year','AQI']).size().unstack(fill_value=0)
    yoy = yoy.reindex(columns=[c for c in AQI_ORDER if c in yoy.columns], fill_value=0)
    fig9, ax9 = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(yoy))
    for cls in yoy.columns:
        ax9.bar(yoy.index.astype(str), yoy[cls], bottom=bottom,
                label=cls, color=AQI_COLOURS.get(cls,"#aaa"))
        bottom += yoy[cls].values
    ax9.set_title(f"AQI Trend 2026-2030 - {fc_city} ({fc_model})")
    ax9.set_xlabel("Year"); ax9.set_ylabel("Days")
    ax9.legend(loc='upper right', fontsize=9)
    plt.tight_layout(); st.pyplot(fig9); plt.close(fig9)

    # All cities comparison
    st.subheader(f"All Cities AQI Comparison - {fc_year} ({fc_model})")
    yr_fc        = forecast[forecast['Year']==fc_year].copy()
    yr_fc['AQI'] = yr_fc[pred_col]
    city_sum     = yr_fc.groupby('Location')['AQI'].agg(
        lambda x: x.value_counts().idxmax()).reset_index()
    city_sum.columns = ['City','Most Common AQI']
    city_sum['Pct']  = yr_fc.groupby('Location')['AQI'].apply(
        lambda x: x.value_counts(normalize=True).max()*100).values
    city_sum = city_sum.sort_values('Pct', ascending=True)
    fig8, ax8 = plt.subplots(figsize=(10, max(4, len(city_sum)*0.5)))
    bar_c = [AQI_COLOURS.get(a,"#aaa") for a in city_sum['Most Common AQI']]
    bars  = ax8.barh(city_sum['City'], city_sum['Pct'], color=bar_c)
    ax8.set_xlabel("% of Days"); ax8.set_title(f"Most Common AQI by City - {fc_year}")
    for bar, (aqi_l, pct) in zip(bars, zip(city_sum['Most Common AQI'], city_sum['Pct'])):
        ax8.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                 f"{aqi_l} ({pct:.0f}%)", va='center', fontsize=9)
    ax8.set_xlim(0, 115)
    plt.tight_layout(); st.pyplot(fig8); plt.close(fig8)

    st.divider()

    # Summary table
    st.subheader(f"AQI Summary - {fc_city} {fc_year}")
    summary = city_fc['AQI'].value_counts().reset_index()
    summary.columns = ['AQI Category','Days']
    summary['% of Total'] = (summary['Days']/len(city_fc)*100).round(1).astype(str)+'%'
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # Downloads
    st.subheader("Download Forecast Data")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        out = city_fc[['Datetime','AQI']].copy()
        out['Datetime'] = pd.to_datetime(out['Datetime']).dt.strftime('%Y-%m-%d')
        out.columns = ['Date','Predicted AQI']
        buf = io.StringIO(); out.to_csv(buf, index=False)
        st.download_button(f"Download {fc_city} {fc_year}", buf.getvalue(),
                           f"AQI_{fc_year}_{fc_city}.csv", "text/csv")
    with dl2:
        yr_out = yr_fc[['Datetime','Location',pred_col]].copy()
        yr_out['Datetime'] = pd.to_datetime(yr_out['Datetime']).dt.strftime('%Y-%m-%d')
        yr_out.columns = ['Date','City','Predicted AQI']
        buf2 = io.StringIO(); yr_out.to_csv(buf2, index=False)
        st.download_button(f"Download All Cities {fc_year}", buf2.getvalue(),
                           f"AQI_{fc_year}_AllCities.csv", "text/csv")
    with dl3:
        full_out = forecast[['Datetime','Year','Location',pred_col]].copy()
        full_out['Datetime'] = pd.to_datetime(full_out['Datetime']).dt.strftime('%Y-%m-%d')
        full_out.columns = ['Date','Year','City','Predicted AQI']
        buf3 = io.StringIO(); full_out.to_csv(buf3, index=False)
        st.download_button("Download Full 2026-2030", buf3.getvalue(),
                           "AQI_2026_2030_AllCities.csv", "text/csv")


# ================================================================
# TAB 3 — MODEL PERFORMANCE
# ================================================================
with tab3:
    st.header("Model Performance Report")
    st.markdown("""
    This dashboard evaluates three machine learning models for UK AQI prediction:
    **Random Forest**, **XGBoost**, and **LSTM**. 
    Models are assessed using Accuracy, Precision, Recall, and F1-Score on a 20% held-out validation set.
    """)

    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Model Accuracy & Metrics Comparison")
        try:
            st.image(plt.imread("model_comparison.png"), use_column_width=True)
        except Exception:
            st.info("Run train_model.py to generate model_comparison.png")

    with p2:
        st.subheader("Confusion Matrix (Best Model)")
        try:
            st.image(plt.imread("confusion_matrix.png"), use_column_width=True)
        except Exception:
            st.info("Run train_model.py to generate confusion_matrix.png")

    st.divider()

    st.subheader("Feature Engineering Summary")
    feat_info = pd.DataFrame({
        "Feature Type":  ["Raw Pollutants", "Raw Pollutants", "Raw Pollutants",
                          "Temporal", "Temporal", "Temporal", "Temporal", "Temporal",
                          "Lag Features", "Rolling Mean",
                          "Location"],
        "Feature":       ["PM2.5", "PM10", "NO2",
                          "Hour", "Day", "Month", "Day of Week", "Season",
                          "Lag 1h/2h/3h/6h/12h/24h per pollutant",
                          "6h and 24h rolling mean per pollutant",
                          "City encoder"],
        "Description":   [
            "Fine particulate matter (ug/m3)",
            "Coarse particulate matter (ug/m3)",
            "Nitrogen dioxide (ug/m3)",
            "Hour of day (0-23)",
            "Day of month (1-31)",
            "Month of year (1-12)",
            "Day of week (0=Mon)",
            "Season (0=Winter to 3=Autumn)",
            "Historical pollutant values up to 24h back",
            "Average pollutant levels over past 6h and 24h",
            "Integer encoding of city name"
        ]
    })
    st.dataframe(feat_info, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("UK DAQI AQI Classification Guide")
    guide = pd.DataFrame({
        "AQI Category":   ["Low", "Moderate", "High", "Very High"],
        "PM2.5 (ug/m3)": ["<= 11", "<= 23", "<= 35", "> 35"],
        "PM10 (ug/m3)":  ["<= 16", "<= 33", "<= 50", "> 50"],
        "NO2 (ug/m3)":   ["<= 67", "<= 134", "<= 200", "> 200"],
        "Health Advice":  [
            "Normal outdoor activities",
            "Sensitive groups take care",
            "Reduce strenuous outdoor activity",
            "Avoid outdoor activity"
        ]
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)

# ================================================================
# FOOTER
# ================================================================
st.divider()
st.caption("UK AQI Research Dashboard | Data: 2021-2026 | Forecast: 2026-2030 | Models: RF | XGBoost | LSTM | UK DAQI Scale")
