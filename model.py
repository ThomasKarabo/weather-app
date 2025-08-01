import pickle
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
import sqlite3



db_path = r'weather.db'
conn = sqlite3.connect(db_path)
query = "SELECT * FROM jhb_weather"
df = pd.read_sql_query(query, conn)
conn.close()

# Ensure the index is datetime and all columns are float
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

def train_var_model(data, order, save_path=None):
    """
    Trains a VAR model on the provided data and optionally saves it to a pickle file.

    Args:
        data (pd.DataFrame): The DataFrame containing the training data.
        order (int): The order of the VAR model.
        save_path (str, optional): The path to save the trained model as a pickle file. Defaults to None.

    Returns:
        statsmodels.tsa.vector_ar_model.VARResultsWrapper: The fitted VAR model.
    """
    try:
        model = VAR(data)

        # Automatically select best lag based on AIC
        results = model.select_order(maxlags=15)
        best_lag = results.aic

        # Fit final model using best lag
        model_fitted = model.fit(best_lag)

        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(model_fitted, f)
            print(f"VAR model trained and saved to {save_path}")

        return model_fitted

    except Exception as e:
        print(f"Error during model training: {e}")
        return None

def predict_with_var_model(model_path, recent_data_input, forecast_steps, last_known_date):
    """
    Loads a trained VAR model and makes future predictions.

    Args:
        model_path (str): The path to the pickled model file.
        recent_data_input (np.ndarray): The last lag_order values from recent data.
        forecast_steps (int): The number of steps to forecast into the future.
        last_known_date (pd.Timestamp): The date of the last data point used for training/input.

    Returns:
        pd.DataFrame: DataFrame containing future predictions, or None if error.
    """
    try:
        with open(model_path, 'rb') as f:
            model_fitted = pickle.load(f)

        # Ensure the input data has the correct shape for the model
        lag_order = model_fitted.k_ar
        # print(f"Expected input shape: {lag_order}, {model_fitted.nobs}") # Debugging
        # print(f"Actual input shape: {recent_data_input.shape[0]}, {recent_data_input.shape[1]}") # Debugging
        if recent_data_input.shape[0] != lag_order:
             print(f"Error: Input data shape mismatch. Expected {lag_order} rows, got {recent_data_input.shape[0]}")
             return None

        future_predictions_values = model_fitted.forecast(y=recent_data_input, steps=forecast_steps)

        # Create a date index for the future period based on the last known date
        future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

        # Use the column names from the fitted model
        predicted_future_df = pd.DataFrame(future_predictions_values, index=future_dates, columns=model_fitted.model.endog_names)
        return predicted_future_df

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
