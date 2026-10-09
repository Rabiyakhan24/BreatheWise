import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="BreatheWise | InnovateX", page_icon="🌿", layout="wide")

st.markdown("""
<style>
/* BreatheWise high-contrast dark theme */
.stApp, [data-testid="stAppViewContainer"] {
    background: #0B1220;
    color: #F8FAFC;
}
[data-testid="stHeader"] {
    background: #0B1220;
}
[data-testid="stSidebar"] {
    background: #111D30;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
.stApp p, .stApp label, .stApp li, .stApp span, .stApp div {
    color: inherit;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #F8FAFC !important;
}
[data-testid="stCaptionContainer"] p,
.small-note {
    color: #C0CDDF !important;
}
[data-testid="stMetric"] {
    background: #142238;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #36506E;
}
[data-testid="stMetricLabel"] {
    color: #D5E2F2 !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}
[data-testid="stMetricDelta"] {
    color: #6EE7B7 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stFileUploader"] section {
    color: #F8FAFC !important;
}
[data-testid="stFileUploader"] section {
    background: #142238;
    border: 1px dashed #5B789B;
}

/* Fix: readable values in number inputs (including simulated IoT sensor readings) */
[data-testid="stNumberInput"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #F8FAFC !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] button {
    color: #111827 !important;
}
[data-testid="stFileUploader"] button {
    color: #0B1220 !important;
    background: #67E8F9 !important;
    border: 0 !important;
}
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    color: #E2F3FF !important;
    background: #101B2C !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #36506E;
}
[data-testid="stAlert"] {
    border-radius: 10px;
}
.stButton button {
    color: #07111F !important;
    background: #67E8F9 !important;
    border: 0;
    font-weight: 600;
}
.stButton button:hover {
    background: #A5F3FC !important;
    color: #07111F !important;
}
/* Make links and interactive controls readable */
.stApp a, [data-testid="stMarkdownContainer"] a {
    color: #67E8F9 !important;
    text-decoration: underline;
}
.stApp a:hover, [data-testid="stMarkdownContainer"] a:hover {
    color: #A5F3FC !important;
}
/* File uploader: visible title, helper text, and browse button */
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * {
    color: #F8FAFC !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    color: #07111F !important;
    background: #67E8F9 !important;
    border: 1px solid #67E8F9 !important;
}
/* Dropdowns and multiselects */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: #142238 !important;
    border-color: #5B789B !important;
    color: #F8FAFC !important;
}
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] * {
    color: #F8FAFC;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
    background: #142238 !important;
    color: #F8FAFC !important;
}
[role="option"] {
    color: #F8FAFC !important;
}
[role="option"]:hover {
    background: #24415F !important;
}
/* Download buttons and secondary buttons */
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button,
button[kind="secondary"] {
    background: #142238 !important;
    color: #F8FAFC !important;
    border: 1px solid #67E8F9 !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover,
button[kind="secondary"]:hover {
    background: #24415F !important;
    color: #FFFFFF !important;
}
/* Expanders and tabs */
[data-testid="stExpander"] summary,
[data-testid="stTabs"] button {
    color: #F8FAFC !important;
}
[data-testid="stExpander"] {
    border-color: #36506E !important;
}
/* Altair/Vega chart toolbar: dark background and visible icons */
.vega-embed .vega-actions,
.vega-embed details,
.vega-embed summary {
    background: #142238 !important;
    border: 1px solid #36506E !important;
    border-radius: 6px !important;
}
.vega-embed .vega-actions button,
.vega-embed .vega-actions a {
    color: #F8FAFC !important;
    background: #142238 !important;
    border: 0 !important;
    opacity: 1 !important;
}
.vega-embed .vega-actions button:hover,
.vega-embed .vega-actions a:hover {
    color: #67E8F9 !important;
    background: #24415F !important;
}
.vega-embed .vega-actions svg,
.vega-embed .vega-actions path {
    fill: #F8FAFC !important;
    stroke: #F8FAFC !important;
}
.vega-embed .vega-actions button:focus,
.vega-embed .vega-actions a:focus {
    outline: 2px solid #67E8F9 !important;
}
hr {
    border-color: #36506E !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 BreatheWise")
st.caption("PREDICT. ACT. BREATHE BETTER.  |  InnovateX · Air Pollution Intelligence")
st.markdown("""
BreatheWise combines **pollution-data analysis, ML forecasting, hotspot prioritisation, and action recommendations**.
This fest prototype uses a dataset you upload. Its IoT panel is a **clearly labelled simulation** until an ESP32 and physical sensors are connected.
""")

DATA_PATH = Path(__file__).parent / "sample_demo_data.csv"
uploaded = st.sidebar.file_uploader("Upload historical pollution CSV", type=["csv"])
st.sidebar.markdown("**Required:** timestamp + PM2.5")
st.sidebar.caption("Optional: PM10, temperature, humidity, wind speed, station/location.")
use_sample = st.sidebar.checkbox("Use included sample dataset when no CSV is uploaded", value=True)
show_iot = st.sidebar.checkbox("Show simulated IoT sensor node (demo only)", value=True)

@st.cache_data
def load_demo():
    return pd.read_csv(DATA_PATH)

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = f"Uploaded file: {uploaded.name}"
    elif use_sample:
        raw = load_demo()
        source_label = "Included synthetic sample dataset"
    else:
        st.info("Upload a CSV from the sidebar to begin.")
        st.stop()
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

def key(v):
    return "".join(ch.lower() for ch in str(v).strip() if ch.isalnum())

original_columns = list(raw.columns)
norm_cols = {key(c): c for c in original_columns}

def find_col(candidates, tokens=()):
    for candidate in candidates:
        if key(candidate) in norm_cols:
            return norm_cols[key(candidate)]
    for norm, original in norm_cols.items():
        if any(token in norm for token in tokens):
            return original
    return None

date_src = find_col(["datetime", "timestamp", "date_time", "date time", "date", "time", "from date", "start date"], ("datetime", "timestamp", "fromdate", "startdate"))
pm25_src = find_col(["pm25", "pm2.5", "pm 2.5", "pm2_5", "pm25 concentration", "pm2.5 (ug/m3)", "pm2.5 (µg/m³)"], ("pm25", "pm2"))
pm10_src = find_col(["pm10", "pm10.0", "pm10 (ug/m3)"], ("pm10",))
temp_src = find_col(["temperature", "temp", "temperature c", "temperature_c"], ("temperature",))
humidity_src = find_col(["humidity", "relative humidity", "humidity pct"], ("humidity",))
wind_src = find_col(["wind speed", "wind_speed", "windspeed"], ("windspeed", "windvelocity"))
station_src = find_col(["station", "station name", "location", "site", "monitoring station", "city"], ("station", "location", "monitoringsite"))

if date_src is None or pm25_src is None:
    st.warning("Please map the required columns manually.")
    c1, c2 = st.columns(2)
    with c1:
        date_src = st.selectbox("Timestamp/date-time column", original_columns, index=0)
    with c2:
        default_target = next((i for i, c in enumerate(original_columns) if "pm" in key(c) and ("25" in key(c) or "2" in key(c))), 0)
        pm25_src = st.selectbox("PM2.5 column", original_columns, index=default_target)
    if date_src == pm25_src:
        st.error("Timestamp and PM2.5 must be different columns.")
        st.stop()

df = raw.copy()
df["datetime"] = pd.to_datetime(df[date_src], errors="coerce", dayfirst=True)
df["pm25"] = pd.to_numeric(df[pm25_src].astype(str).str.replace(",", "", regex=False), errors="coerce")
if pm10_src:
    df["pm10"] = pd.to_numeric(df[pm10_src], errors="coerce")
if temp_src:
    df["temperature"] = pd.to_numeric(df[temp_src], errors="coerce")
if humidity_src:
    df["humidity"] = pd.to_numeric(df[humidity_src], errors="coerce")
if wind_src:
    df["wind_speed"] = pd.to_numeric(df[wind_src], errors="coerce")
if station_src:
    df["station_name"] = df[station_src].astype(str).fillna("Unknown")

df = df.dropna(subset=["datetime", "pm25"])
if df.empty:
    st.error("No valid timestamp + PM2.5 rows were found.")
    st.stop()

# Station filter: keep station histories separate rather than mixing different sites.
if "station_name" in df.columns:
    station_options = sorted(df["station_name"].dropna().astype(str).unique().tolist())
    if len(station_options) > 1:
        selected_station = st.sidebar.selectbox("Model station", ["All stations (not recommended for single-site forecast)"] + station_options, index=1)
        if selected_station != "All stations (not recommended for single-site forecast)":
            df = df[df["station_name"] == selected_station].copy()
        else:
            st.warning("Multiple stations selected together. For a reliable single-location forecast, select one station in the sidebar.")
    else:
        selected_station = station_options[0] if station_options else "Unknown"
else:
    selected_station = "Not provided in dataset"

# Duplicate timestamps can occur across stations; after optional filtering, keep latest row per timestamp.
df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
if len(df) < 30:
    st.error(f"Only {len(df)} valid rows remain. Upload at least 30 time-stamped observations for this prototype.")
    st.stop()

median_gap = df["datetime"].diff().dropna().dt.total_seconds().median() / 3600
if pd.isna(median_gap) or median_gap <= 0:
    median_gap = 1.0

st.success(f"{source_label} · {len(df):,} valid rows · detected interval ≈ {median_gap:g} hours.")
with st.expander("Dataset quality and column mapping"):
    quality_cols = [c for c in ["datetime", "pm25", "pm10", "temperature", "humidity", "wind_speed", "station_name"] if c in df.columns]
    st.write({"Timestamp source": str(date_src), "PM2.5 source": str(pm25_src), "Station source": str(station_src) if station_src else "Not available", "Median sampling interval (hours)": round(float(median_gap), 3), "Selected station": str(selected_station)})
    st.dataframe(pd.DataFrame({"column": quality_cols, "missing_values": [int(df[c].isna().sum()) for c in quality_cols]}), use_container_width=True)

# Use a per-observation forecast target; actual horizon depends on measured sampling interval.
df["lag1"] = df["pm25"]
df["lag2"] = df["pm25"].shift(1)
df["lag3"] = df["pm25"].shift(2)
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["target_next"] = df["pm25"].shift(-1)

feature_cols = ["lag1", "lag2", "lag3", "hour", "day_of_week", "month"]
# Current observation's covariates can help predict the next observation.
for c in ["pm10", "temperature", "humidity", "wind_speed"]:
    if c in df.columns and df[c].notna().mean() >= 0.8:
        feature_cols.append(c)

model_df = df.dropna(subset=feature_cols + ["target_next"]).copy()
if len(model_df) < 25:
    st.error("Not enough complete rows after creating lag features. Check missing values and column mapping.")
    st.stop()

split = int(len(model_df) * 0.8)
split = min(max(split, 15), len(model_df) - 5)
train, test = model_df.iloc[:split], model_df.iloc[split:]
X_train, y_train = train[feature_cols], train["target_next"]
X_test, y_test = test[feature_cols], test["target_next"]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42)
}
model_results = {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    pred_test = estimator.predict(X_test)
    model_results[name] = {
        "model": estimator,
        "pred": np.maximum(0, pred_test),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": float(mean_squared_error(y_test, pred_test) ** 0.5)
    }

comparison = pd.DataFrame([{"Model": name, "MAE (µg/m³)": r["mae"], "RMSE (µg/m³)": r["rmse"]} for name, r in model_results.items()])
best_name = comparison.sort_values("MAE (µg/m³)").iloc[0]["Model"]
chosen_name = st.sidebar.selectbox("Forecast model", list(models.keys()), index=list(models.keys()).index("Random Forest"))
chosen = model_results[chosen_name]
best_model = model_results[best_name]
latest = df.iloc[-1]
latest_dt = latest["datetime"]

# Forecast one next observation from the last known feature row. For 3-step forecast, recursive PM2.5
# lags are used and the latest available weather/PM10 covariates are held constant.
last_valid = df.dropna(subset=["pm25"]).iloc[-1]
previous_rows = df.dropna(subset=["pm25"]).tail(3)
if len(previous_rows) < 3:
    st.error("Need at least 3 PM2.5 observations for forecasting.")
    st.stop()

def build_xnext(history_values, future_time, last_row):
    vals = list(history_values)
    payload = {
        "lag1": float(vals[-1]),
        "lag2": float(vals[-2]),
        "lag3": float(vals[-3]),
        "hour": int(future_time.hour),
        "day_of_week": int(future_time.dayofweek),
        "month": int(future_time.month)
    }
    for c in ["pm10", "temperature", "humidity", "wind_speed"]:
        if c in feature_cols:
            v = pd.to_numeric(pd.Series([last_row.get(c, np.nan)]), errors="coerce").iloc[0]
            payload[c] = float(v) if pd.notna(v) else 0.0
    return pd.DataFrame([[payload[c] for c in feature_cols]], columns=feature_cols)

last_three = previous_rows["pm25"].astype(float).tolist()
delta = pd.to_timedelta(float(median_gap), unit="h")
future_time_1 = latest_dt + delta
x1 = build_xnext(last_three, future_time_1, latest)
forecast_1 = max(0.0, float(chosen["model"].predict(x1)[0]))
future_time_3 = latest_dt + pd.to_timedelta(float(median_gap) * 3, unit="h")
history_vals = last_three.copy()
recursive_vals = []
for step in range(1, 4):
    ft = latest_dt + delta * step
    xfuture = build_xnext(history_vals, ft, latest)
    value = max(0.0, float(chosen["model"].predict(xfuture)[0]))
    recursive_vals.append(value)
    history_vals.append(value)
forecast_3 = recursive_vals[-1]

# High-contrast summary
st.divider()
st.subheader("01 · Monitor your dataset")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest PM2.5", f"{float(latest['pm25']):.1f} µg/m³")
m2.metric(f"Forecast · +{median_gap:g}h", f"{forecast_1:.1f} µg/m³", f"{forecast_1-float(latest['pm25']):+.1f} vs latest")
m3.metric(f"Forecast · +{median_gap*3:g}h (recursive)", f"{forecast_3:.1f} µg/m³")
m4.metric("Test MAE · selected model", f"{chosen['mae']:.2f} µg/m³")
st.caption(f"Station: {selected_station} · Data window: {df['datetime'].min():%d %b %Y %H:%M} – {latest_dt:%d %b %Y %H:%M} · Source: {source_label}. Forecast horizon follows detected sampling interval, not an assumed hour.")

# Trend and action rules
left, right = st.columns([1.55, 1])
with left:
    st.subheader("Historical PM2.5 trend")
    trend = df.tail(168)[["datetime", "pm25"]].copy()
    chart = alt.Chart(trend).mark_line(color="#67E8F9", strokeWidth=2.5).encode(
        x=alt.X("datetime:T", title="Timestamp"),
        y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("datetime:T", title="Timestamp"), alt.Tooltip("pm25:Q", title="PM2.5", format=".2f")]
    ).properties(height=300, background="#0B1220")
    st.altair_chart(chart, use_container_width=True, theme=None)
with right:
    st.subheader("Action recommendation")
    if forecast_1 >= 90:
        st.error("HIGH PRIORITY · Forecast is elevated. Review local conditions, reduce avoidable emissions where feasible, and follow official health guidance.")
        action = "Prioritise source inspection, dust suppression at active construction sites, and traffic-flow measures where locally appropriate."
    elif forecast_1 >= 60:
        st.warning("WATCH · Forecast crosses the prototype elevated threshold.")
        action = "Increase monitoring frequency; consider road-dust management and avoiding unnecessary idling near busy roads."
    else:
        st.success("MONITOR · Forecast is below the prototype alert threshold.")
        action = "Continue monitoring trends and maintain routine dust, waste-burning and vehicle-emission prevention."
    st.markdown(f"**Suggested response**\n\n{action}")
    st.caption("Rule-based decision support only. Thresholds are illustrative, not official AQI categories or medical advice.")

st.divider()
st.subheader("02 · Compare ML algorithms")
st.caption("All models use the same chronological train/test split and feature set. Lowest test MAE is best on this dataset; it does not guarantee future performance.")
st.dataframe(comparison.sort_values("MAE (µg/m³)").style.format({"MAE (µg/m³)": "{:.2f}", "RMSE (µg/m³)": "{:.2f}"}), use_container_width=True, hide_index=True)
best_cols = st.columns(3)
best_cols[0].metric("Best test MAE in this run", best_name)
best_cols[1].metric("Training rows", f"{len(train):,}")
best_cols[2].metric("Held-out test rows", f"{len(test):,}")
eval_df = pd.DataFrame({
    "datetime": pd.to_datetime(test["datetime"], errors="coerce").to_numpy(),
    "Actual PM25": pd.to_numeric(y_test, errors="coerce").to_numpy(),
    "Predicted PM25": pd.to_numeric(chosen["pred"], errors="coerce")
}).replace([np.inf, -np.inf], np.nan).dropna().tail(120)

if not eval_df.empty:
    eval_long = eval_df.melt("datetime", var_name="Series", value_name="Concentration")
    eval_chart = alt.Chart(eval_long).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("datetime:T", title="Timestamp",
                axis=alt.Axis(labelColor="#D5E2F2", titleColor="#D5E2F2",
                              gridColor="#26364B", labelOverlap=True, format="%d %b")),
        y=alt.Y("Concentration:Q", title="PM2.5 (µg/m³)",
                axis=alt.Axis(labelColor="#D5E2F2", titleColor="#D5E2F2",
                              gridColor="#26364B"),
                scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N",
                        scale=alt.Scale(domain=["Actual PM25", "Predicted PM25"],
                                        range=["#67E8F9", "#FBBF24"]),
                        legend=alt.Legend(title="Series", orient="bottom",
                                          labelColor="#F8FAFC", titleColor="#F8FAFC")),
        tooltip=[
            alt.Tooltip("datetime:T", title="Timestamp"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Concentration:Q", title="PM2.5", format=".2f")
        ]
    ).properties(height=360, background="#0B1220", padding={"left": 10, "right": 10, "top": 10, "bottom": 10}).configure_view(
        stroke="#36506E"
    ).configure_axis(domainColor="#7890AA", tickColor="#7890AA")
    st.altair_chart(eval_chart, use_container_width=True, theme=None)
    st.caption("Cyan = actual next-observation PM2.5; yellow = selected model prediction on the held-out test period.")

# Station hotspot comparison if location/station data exists. If no station column, say so—don't invent geography.
st.divider()
st.subheader("03 · Hotspot prioritisation")
if station_src and station_src in raw.columns:
    hot = raw.copy()
    hot["__time"] = pd.to_datetime(hot[date_src], errors="coerce", dayfirst=True)
    hot["__pm25"] = pd.to_numeric(hot[pm25_src].astype(str).str.replace(",", "", regex=False), errors="coerce")
    hot["__station"] = hot[station_src].astype(str)
    hot = hot.dropna(subset=["__time", "__pm25"])
    if not hot.empty:
        latest_per_station = hot.sort_values("__time").groupby("__station", as_index=False).tail(1)
        latest_per_station = latest_per_station.sort_values("__pm25", ascending=False)
        st.caption("Latest available PM2.5 reading per station in the uploaded file. This is a station comparison, not a geographic map.")
        st.dataframe(latest_per_station[["__station", "__time", "__pm25"]].rename(columns={"__station":"Station", "__time":"Latest timestamp", "__pm25":"Latest PM2.5 (µg/m³)"}), use_container_width=True, hide_index=True)
        st.bar_chart(latest_per_station.set_index("__station")["__pm25"], use_container_width=True)
else:
    st.info("No station/location column was found. Upload data containing a station or location field to compare real locations. The prototype will not invent hotspot names.")

# Simulated IoT module, deliberately separate from the historical dataset and forecasts.
st.divider()
st.subheader("04 · IoT sensor node")
if show_iot:
    st.warning("SIMULATED IOT DEMO — no ESP32 or physical sensor is connected. Values below are illustrative and must not be described as live measurements.")
    with st.container(border=True):
        st.markdown("**Virtual ESP32 + particulate sensor**")
        i1, i2, i3, i4 = st.columns(4)
        default_pm25 = float(np.clip(latest["pm25"], 0, 500))
        default_pm10 = float(np.clip(latest["pm10"], 0, 800)) if "pm10" in df.columns and pd.notna(latest.get("pm10", np.nan)) else float(np.clip(default_pm25*1.5, 0, 800))
        default_temp = float(latest["temperature"]) if "temperature" in df.columns and pd.notna(latest.get("temperature", np.nan)) else 28.0
        default_hum = float(latest["humidity"]) if "humidity" in df.columns and pd.notna(latest.get("humidity", np.nan)) else 55.0
        virtual_pm25 = i1.number_input("Simulated PM2.5 (µg/m³)", 0.0, 500.0, default_pm25, 1.0)
        virtual_pm10 = i2.number_input("Simulated PM10 (µg/m³)", 0.0, 800.0, default_pm10, 1.0)
        virtual_temp = i3.number_input("Simulated temperature (°C)", -10.0, 60.0, default_temp, 0.5)
        virtual_hum = i4.number_input("Simulated humidity (%)", 0.0, 100.0, default_hum, 1.0)
        st.caption("These controls let you demonstrate the sensor-to-dashboard flow. They do not send data to a real ESP32 or retrain the model automatically.")
        if virtual_pm25 >= 90:
            st.error("Virtual sensor alert: high particulate concentration in the demo scenario.")
        elif virtual_pm25 >= 60:
            st.warning("Virtual sensor alert: elevated particulate concentration in the demo scenario.")
        else:
            st.success("Virtual sensor status: below the demo alert threshold.")
else:
    st.info("IoT demo panel hidden. Connect a real ESP32 + PM sensor before claiming physical sensor integration.")

with st.expander("Latest data records"):
    st.dataframe(df.tail(20), use_container_width=True)
with st.expander("Architecture and limitations"):
    st.markdown("""
    **Architecture:** PM sensor → ESP32 → Wi-Fi/HTTP ingestion → timestamped storage → ML forecast → rule-based action recommendation.

    **Current prototype status**
    - CSV upload and ML training: implemented.
    - Linear Regression, Random Forest, HistGradientBoosting comparison: implemented.
    - 1-step and recursive 3-step PM2.5 forecast: implemented; the time horizon depends on the detected sampling interval.
    - Station-level hotspot comparison: available only if the uploaded CSV has station/location names.
    - IoT panel: simulation only; no physical hardware is connected.
    - Action recommendations: rule-based prototype; actual pollution reduction has not been measured.

    **Validation:** chronological 80/20 split; report test MAE/RMSE. The 3-step forecast recursively uses prior predicted PM2.5 and holds optional weather covariates at the latest observed values, so treat it as exploratory until validated.
    """)
st.download_button("Download held-out predictions CSV", data=eval_df.to_csv(index=False).encode("utf-8") if not eval_df.empty else b"", file_name="breathewise_test_predictions.csv", mime="text/csv", disabled=eval_df.empty)
