import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from google.cloud import storage
import joblib
import io
import json

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(series, eval_year=None):
    """Train the model and evaluate its performance."""
    # Diviser les données en train et test
    train, test = series[:str(eval_year)], series[str(eval_year):]
    
    # Entraîner le modèle sur les données d'entraînement
    model = get_best_arima_model(train)
    
    # Prédire les valeurs sur les données de test
    y_pred = model.predict(n_periods=len(test))
    
    # Calculer l'RMSE
    accuracy = calculate_rmse(test.values, y_pred)
    
    # Retourner l'accuracy et les paramètres du modèle
    return accuracy, model.order, model.seasonal_order

def get_best_arima_model(series):
    model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Meilleure configuration trouvée : ordre={best_order}, ordre saisonnier={best_seasonal_order}")
    return model

def calculate_accuracy(series, model):
    if len(series) == 0:
        return 0 
    predictions = model.predict_in_sample()
    actual = series.values
    if len(predictions) != len(actual):
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
    accuracy = 100 - np.mean(np.abs((actual - predictions) / actual)) * 100
    return accuracy

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

def generate_monitoring_plot(code, entity_type, accuracies, bucket_name, blob_name):
    epochs = list(range(1, len(accuracies) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, accuracies, marker='o', label='Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
    ax.legend()
    ax.grid(True)
    
    save_plot_to_gcs(fig, bucket_name, blob_name)

def save_model_info_to_gcs(model_info, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    model_info_json = json.dumps(model_info)
    blob.upload_from_string(model_info_json, content_type='application/json')
    print(f"Model info uploaded to {blob_name}.")

def load_model_info_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    model_info_json = blob.download_as_string()
    model_info = json.loads(model_info_json)
    print(f"Model info loaded from {blob_name}.")
    return model_info
