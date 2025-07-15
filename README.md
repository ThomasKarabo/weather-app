# 🌦️ Johannesburg Weather Forecast App

A full-stack data science project that fetches, stores, models, and forecasts weather data for Johannesburg using Open-Meteo API, SQLite, VAR models, and a Streamlit dashboard.

---

## 📊 Project Overview

This app:
- Pulls **historical** and **real-time weather data** for Johannesburg from the [Open-Meteo API](https://open-meteo.com).
- Stores the data in a **local SQLite database**.
- Trains a **Vector AutoRegressive (VAR)** model to forecast multiple weather features.
- Displays current, past, and **7-day weather forecasts** in an interactive **Streamlit dashboard**.

---

## 🚀 Features

✅ Real-time **current weather**  
✅ Historical data collection (since 2005)  
✅ **Automated ETL** using FastAPI  
✅ 7-day weather forecast using **VAR** model  
✅ Visualisation of key weather metrics  
✅ Deployable on **Streamlit Cloud**

---

## 🏗️ Architecture

```mermaid
A[FastAPI] --> B[SQLite Database]
B --> C[Model Trainer (VAR)]
C --> D[Pickled Model]
B --> E[Streamlit App]
D --> E
```
## 🔧 Tech Stack
Layer	Tools
API	FastAPI
Database	SQLite
Data Access	Open-Meteo API
ML Model	VAR (via statsmodels)
Frontend	Streamlit
Deployment	Streamlit Cloud
Data Handling	pandas, numpy

## 🧠 Model Info
The project uses a Vector AutoRegressive (VAR) model to predict the following weather features over the next 7 days:

Max & Min Temperature

Wind Speed

Wind Direction

Precipitation

## 🗂️ File Structure
```
weather-app/
├── app.py               # Streamlit dashboard
├── model.py             # VAR model training + prediction logic
├── weather_api.py       # FastAPI ETL to fetch and save data
├── weather.db           # SQLite database (created on first run)
├── var_model_latest.pkl # Trained VAR model (auto-generated)
├── requirements.txt     # Python dependencies
└── README.md            # You're here 😎
```

## 🚦 How to Run Locally
1. Clone the repo
```
git clone https://github.com/your-username/weather-app.git
cd weather-app
```
2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the FastAPI ETL
```
uvicorn weather_api:app --reload
```
4. Run the Streamlit App
```
streamlit run app.py
```
## ☁️ Streamlit Cloud Deployment
The app is also deployed on Streamlit Cloud:

👉 [Live Demo](https://weather-app-aq5opgjg3fesucfkd4kkbu.streamlit.app)

## 📈 Future Improvements
Add more ML models (e.g. LSTM, XGBoost) for comparison

Store & analyse hourly weather data

Add user input for custom city selection

Deploy the FastAPI backend separately with a scheduler or use Airflow to automate the data pull

## 🧑‍💻 Author
Thomas Karabo Mohlapo
📍 South Africa | ☁️ Weather & Data Enthusiast
LinkedIn | GitHub

