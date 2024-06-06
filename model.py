import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from google.cloud import storage
import joblib
import io

def save_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    try:
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"Failed to upload file {source_file_name} to {destination_blob_name}: {e}")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    try:
        blob.download_to_filename(destination_file_name)
        print(f"File {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        print(f"Failed to download file {source_blob_name} to {destination_file_name}: {e}")

def calculate_metrics(y_true, y_pred):
    """Calcule les métriques RMSE, MAE et R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def perform_cross_validation(series, n_splits=10):
    """Effectue une validation croisée et retourne les résultats pour chaque split."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_index, test_index in tscv.split(series):
        train, test = series.iloc[train_index], series.iloc[test_index]
        model = get_best_arima_model(train)
        y_pred = model.predict(n_periods=len(test))
        
        metrics = calculate_metrics(test.values, y_pred)
        
        split_result = {
            "train_index": train_index.tolist(),
            "test_index": test_index.tolist(),
            "train_values": train.tolist(),
            "test_values": test.tolist(),
            "predicted_values": y_pred.tolist(),
            "metrics": metrics,
            "model_order": model.order,
            "model_seasonal_order": model.seasonal_order
        }
        results.append(split_result)

    return results

def get_best_arima_model(series):
    model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Meilleure configuration trouvée : ordre={best_order}, ordre saisonnier={best_seasonal_order}")
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

def generate_monitoring_plot(code, entity_type, cross_val_results, bucket_name, blob_name):
    epochs = list(range(1, len(cross_val_results) + 1))
    rmses = [result['metrics']['rmse'] for result in cross_val_results]
    maes = [result['metrics']['mae'] for result in cross_val_results]
    r2s = [result['metrics']['r2'] for result in cross_val_results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot des métriques
    ax1.plot(epochs, rmses, marker='o', label='RMSE')
    ax1.plot(epochs, maes, marker='o', label='MAE')
    ax1.plot(epochs, r2s, marker='o', label='R²')
    ax1.set_ylabel('Metrics')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title('Métriques de Performance')

    # Plot des valeurs réelles
    ax2.set_ylabel('Population')
    for result in cross_val_results:
        train_values = result['train_values']
        test_values = result['test_values']
        ax2.plot(range(len(train_values), len(train_values) + len(test_values)), test_values, 'b-', alpha=0.6)
    ax2.legend(['Real Population'], loc='upper right')
    ax2.grid(True)
    ax2.set_title('Valeurs Réelles de la Population')

    # Plot des valeurs prédites
    ax3.set_xlabel('Splits')
    ax3.set_ylabel('Population')
    for result in cross_val_results:
        train_values = result['train_values']
        predicted_values = result['predicted_values']
        ax3.plot(range(len(train_values), len(train_values) + len(predicted_values)), predicted_values, 'r-', alpha=0.6)
    ax3.legend(['Predicted Population'], loc='upper right')
    ax3.grid(True)
    ax3.set_title('Valeurs Prédites de la Population')

    plt.suptitle(f'Monitoring des Performances du Modèle pour {entity_type} {code}')

    save_plot_to_gcs(fig, bucket_name, blob_name)

def save_plot_to_gcs(fig, bucket_name, blob_name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    try:
        blob.upload_from_file(buf, content_type='image/png')
        print(f"File {blob_name} uploaded to {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload plot to {bucket_name}/{blob_name}: {e}")
    finally:
        buf.close()
        plt.close(fig)
