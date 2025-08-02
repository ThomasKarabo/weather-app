import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
from model import predict_with_var_model
from run_pipeline import pipeline
import sqlite3
import numpy as np
import os

# Run the FastAPI and train the model
pipeline()

# Calculate yesterday's date
now = datetime.now()
yesterday = now - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

st.title("Johannesburg Weather App ⛅")

# --- Today's Weather Section ---
today_str = now.strftime("%Y-%m-%d")
TODAY_API_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude=-26.1135&longitude=28.0666"
    "&current_weather=true"
    "&hourly=temperature_2m,relativehumidity_2m,precipitation"
    "&timezone=auto"
)

st.header("Today's Weather 😊")

try:
    response = requests.get(TODAY_API_URL)
    response.raise_for_status()
    data = response.json()
    
    # Get current weather data
    current_weather = data.get("current_weather", {})
    hourly_data = data.get("hourly", {})
    
    # Get the most recent hourly data (index 0 is current hour)
    temp = current_weather.get("temperature")
    windspeed = current_weather.get("windspeed")
    winddir = current_weather.get("winddirection")
    
    # Get additional data from hourly
    if len(hourly_data.get("time", [])) > 0:
        latest_time = hourly_data["time"][0]
        humidity = hourly_data["relativehumidity_2m"][0] if "relativehumidity_2m" in hourly_data else "N/A"
        precip = hourly_data["precipitation"][0] if "precipitation" in hourly_data else "N/A"
    else:
        latest_time = current_weather.get("time")
        humidity = "N/A"
        precip = "N/A"

    if temp is not None:
        st.subheader(f"Latest Weather Data for Today ({latest_time})")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Temperature (°C)", f"{temp}")
        col2.metric("Windspeed (km/h)", f"{windspeed}")
        col3.metric("Wind Direction (°)", f"{winddir}")
        col4.metric("Precipitation (mm)", f"{precip}")
        col5.metric("Humidity (%)", f"{humidity}")
    else:
        st.info("No weather data available for today yet.")
except Exception as e:
    st.error(f"Failed to fetch today's weather data: {e}")
    
# --- 7-Day Forecast Section ---
st.header("7-Day Weather Forecast (VAR Model)")
try:
    
    # Load the latest data from the database
    db_path = os.path.join(os.getcwd(), "weather.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM jhb_weather", conn)
    conn.close()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    full_data_today = df.dropna()
    # Load the model and make a forecast
    from model import train_var_model
    trained_model_today = train_var_model(full_data_today, order=5, save_path=None)
    if trained_model_today:
        lag_order_latest = trained_model_today.k_ar
        recent_data_for_forecast = full_data_today.values[-lag_order_latest:]
        last_date_in_data = full_data_today.index[-1]
        forecast_steps_production = 7
        forecast_df = predict_with_var_model(
            model_path='var_model_latest.pkl',
            recent_data_input=recent_data_for_forecast,
            forecast_steps=forecast_steps_production,
            last_known_date=last_date_in_data
        )
        if forecast_df is not None:
            # Only plot temp_max and temp_min if they exist in the forecast
            cols_to_plot = [col for col in ['temp_max', 'temp_min'] if col in forecast_df.columns]
            if cols_to_plot:
                st.line_chart(forecast_df[cols_to_plot])
            else:
                st.info("Forecast does not contain temp_max or temp_min columns.")
        else:
            st.info("No forecast available.")
    else:
        st.info("Model training failed. No forecast available.")
except Exception as e:
    st.error(f"Failed to generate 7-day forecast: {e}")

# --- Yesterday's Weather Section ---
YESTERDAY_API_URL = (
    f"https://api.open-meteo.com/v1/forecast?latitude=-26.1135&longitude=28.0666"
    f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
    f"&start_date={yesterday_str}&end_date={yesterday_str}"
    f"&timezone=auto"
)

st.header("Yesterday's Weather")
try:
    response = requests.get(YESTERDAY_API_URL)
    response.raise_for_status()
    data = response.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = int(hourly.get("temperature_2m", []))
    windspeeds = int(hourly.get("windspeed_10m", []))
    winddirs = int(hourly.get("winddirection_10m", []))

    # Prepare data for table
    display_hours = ["00:00", "06:00", "12:00", "18:00"]
    hour_labels = [f"{hour}" for hour in display_hours]
    temp_row = []
    windspeed_row = []
    winddir_row = []
    for hour in display_hours:
        dt_str = f"{yesterday_str}T{hour}"
        if dt_str in times:
            idx = times.index(dt_str)
            temp_row.append(temps[idx] if idx < len(temps) else None)
            windspeed_row.append(windspeeds[idx] if idx < len(windspeeds) else None)
            winddir_row.append(winddirs[idx] if idx < len(winddirs) else None)
        else:
            temp_row.append(None)
            windspeed_row.append(None)
            winddir_row.append(None)

    table = pd.DataFrame({
        hour_labels[0]: [temp_row[0], windspeed_row[0], winddir_row[0]],
        hour_labels[1]: [temp_row[1), windspeed_row[1], winddir_row[1]],
        hour_labels[2]: [temp_row[2], windspeed_row[2], winddir_row[2]],
        hour_labels[3]: [temp_row[3], windspeed_row[3], winddir_row[3]],
    }, index=["Temperature (°C)", "Windspeed (km/h)", "Wind Direction (°)"])
    st.table(table)
except Exception as e:
    st.error(f"Failed to fetch yesterday's weather data: {e}")


