from weather_api import startup_event
from model import train_var_model
import sqlite3
import pandas as pd
import pickle

def pipeline():
    # Run the ETL to update weather.db
    startup_event()

    # Load updated data
    conn = sqlite3.connect("weather.db")
    df = pd.read_sql_query("SELECT * FROM jhb_weather", conn)
    conn.close()

    # Preprocess
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    # Train VAR model
    model = train_var_model(df, order=5, save_path="var_model_latest.pkl")
    print("Pipeline completed successfully!")
