# Weather App

This project is a Streamlit application that displays current and historical weather data for Johannesburg, South Africa, using the Open-Meteo API. It also trains and serves a VAR (Vector Autoregression) model for weather forecasting.

## Features
- Fetches and stores historical weather data (from 2005 to recent date) in a local SQLite database.
- Trains a VAR model on the historical data for multi-day weather forecasting.
- Displays current weather (temperature, windspeed, wind direction, precipitation, humidity) using the Open-Meteo API.
- Shows yesterday's weather in a table format.
- Visualizes 7-day weather forecasts (max/min temperature) using a trained VAR model.

## How to Run
1. Make sure you have Python 3.8+ installed.
2. Install dependencies:
   ```sh
   pip install streamlit requests pandas numpy scikit-learn statsmodels fastapi
   ```
3. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```
4. (Optional) Start the FastAPI backend for database and API access:
   ```sh
   uvicorn weather_api:app --reload
   ```

## Data & Model
- Historical weather data is pulled from Open-Meteo and stored in `weather.db` (SQLite).
- The VAR model is trained on the stored data and used for 7-day forecasts.

## API Used
- [Open-Meteo API](https://open-meteo.com/)

---
