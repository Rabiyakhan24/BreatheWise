# BreatheWise — InnovateX Prototype

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt in the folder containing `app.py`.
3. Run:
   `python -m pip install -r requirements.txt`
   `python -m streamlit run app.py`

## What the prototype demonstrates
- Upload a historical CSV with timestamp and PM2.5 columns.
- Automatically detects common timestamp, PM2.5, PM10, temperature, humidity, wind, and station/location headers.
- Compares Linear Regression, Random Forest Regressor, and HistGradientBoostingRegressor using the same chronological 80/20 train/test split.
- Reports held-out MAE/RMSE.
- Forecasts the next observation and a recursive 3-step estimate. The time horizon depends on the detected sampling interval.
- Shows station-level latest PM2.5 comparisons when the input CSV contains station/location names.
- Gives threshold-based action recommendations.
- Includes a clearly labelled simulated IoT panel for demo purposes.

## Important honesty notes
- `sample_demo_data.csv` is synthetic and is not real Delhi data.
- No ESP32 or physical sensor is connected. The IoT panel is a simulation only.
- Hotspots are shown only when station/location names exist in the supplied data; the app does not invent geographic readings.
- PM2.5 concentration is not the same as official AQI.
- The recursive 3-step forecast is exploratory; optional weather variables are held at the latest observed values. Validate on appropriate real data before making accuracy claims.
- Actions are recommendations; actual pollution reduction has not been measured.

## Recommended CSV columns
Required: `datetime` (or timestamp/date) and `pm25` (or PM2.5).
Optional: `pm10`, `temperature`, `humidity`, `wind_speed`, `station` or `location`.
For a real one-hour forecast, data should have consistent hourly timestamps.
