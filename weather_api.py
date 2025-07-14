from fastapi import FastAPI
import requests
import sqlite3
from datetime import datetime, timedelta

# Calculate yesterday's date
now = datetime.now()
yesterday = now - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

app = FastAPI()

DB_PATH = "weather.db"
TABLE_NAME = "jhb_weather"

API_URL = f'https://archive-api.open-meteo.com/v1/archive?latitude=-26.2&longitude=28.0&start_date=2005-01-01&end_date={yesterday_str}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant&timezone=Africa%2FJohannesburg'

def create_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            date TEXT PRIMARY KEY,
            temp_max REAL,
            temp_min REAL,
            precipitation REAL,
            windspeed_max REAL,
            winddirection_dominant REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_weather_data(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for i in range(len(data['daily']['time'])):
        c.execute(f'''
            INSERT OR REPLACE INTO {TABLE_NAME} (date, temp_max, temp_min, precipitation, windspeed_max, winddirection_dominant)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['daily']['time'][i],
            data['daily']['temperature_2m_max'][i],
            data['daily']['temperature_2m_min'][i],
            data['daily']['precipitation_sum'][i],
            data['daily']['windspeed_10m_max'][i],
            data['daily']['winddirection_10m_dominant'][i]
        ))
    conn.commit()
    conn.close()


# Utility to get the latest date in the DB
def get_latest_db_date():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT MAX(date) FROM {TABLE_NAME}")
    result = c.fetchone()
    conn.close()
    return result[0] if result and result[0] else None

@app.on_event("startup")
def startup_event():
    create_table()
    latest_db_date = get_latest_db_date()
    # If DB is empty, fetch from 2005-01-01 to yesterday
    if not latest_db_date:
        fetch_start = "2005-01-01"
    else:
        # Start from the day after the latest date in DB
        fetch_start = (datetime.strptime(latest_db_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end = yesterday_str
    if fetch_start <= fetch_end:
        api_url = f'https://archive-api.open-meteo.com/v1/archive?latitude=-26.2&longitude=28.0&start_date={fetch_start}&end_date={fetch_end}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant&timezone=Africa%2FJohannesburg'
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        insert_weather_data(data)

@app.get("/weather/{date}")
def get_weather(date: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT * FROM {TABLE_NAME} WHERE date = ?", (date,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "date": row[0],
            "temp_max": row[1],
            "temp_min": row[2],
            "precipitation": row[3],
            "windspeed_max": row[4],
            "winddirection_dominant": row[5]
        }
    return {"error": "No data for this date"}
