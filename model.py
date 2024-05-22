import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from google.cloud import storage
import joblib
import io

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """Calculates the Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true, y_pred)

def train_and_evaluate(series, eval_year=None):
    """Train the model and evaluate its performance."""
    # Split data into train and test
    train, test = series[:eval_year], series[eval_year:]
    
    # Train the model on the training data
    model = get_best_arima_model(train)
    
    # Predict the values on the test data
    y_pred = model.predict(n_periods=len(test))
    
    # Calculate RMSE and MAE
    rmse = calculate_rmse(test.values, y_pred)
    mae = calculate_mae(test.values, y_pred)
    
    # Return RMSE, MAE, and model parameters
    return rmse, mae, model.order, model.seasonal_order

def get_best_arima_model(series):
    model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Best configuration found: order={best_order}, seasonal_order={best_seasonal_order}")
    return model

def plot_population_forecast(series, forecast_df, bucket_name, blob_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series, label='Historical Population')
    ax.plot(forecast_df['mean'], label='Forecasted Population')
    if 'mean_ci_lower' in forecast_df.columns and 'mean_ci_upper' in forecast_df.columns:
        ax.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')
    ax.set_title('Population Forecast')
    ax.legend()
    ax.grid(True)
    
    save_plot_to_gcs(fig, bucket_name, blob_name)

def save_plot_to_gcs(fig, bucket_name, blob_name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buf, content_type='image/png')
    
    buf.close()
    plt.close(fig)
    print(f"File {blob_name} uploaded to {bucket_name}.")

def generate_monitoring_plot(code, entity_type, rmse_values, mae_values, bucket_name, blob_name):
    epochs = list(range(1, len(rmse_values) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, rmse_values, marker='o', label='RMSE')
    ax.plot(epochs, mae_values, marker='o', label='MAE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.set_title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
    ax.legend()
    ax.grid(True)
    
    save_plot_to_gcs(fig, bucket_name, blob_name)
