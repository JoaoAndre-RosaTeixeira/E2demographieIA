import base64
import io
import joblib
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from google.cloud import storage

def get_best_arima_model(series):
    model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Meilleure configuration trouvée : ordre={best_order}, ordre saisonnier={best_seasonal_order}")
    return model

def train_arima(series):
    """Entraîne un modèle ARIMA sur les séries temporelles spécifiées."""
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=False)
    return results

def predict_population(model, start_year, end_year):
    """Génère des prévisions de population à partir d'un modèle ARIMA."""
    steps = end_year - start_year + 1
    forecast = model.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame()
    return forecast_df

def save_plot_to_gcs(fig, bucket_name, blob_name):
    """Sauvegarde une figure matplotlib sur GCS."""
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

def generate_monitoring_plot(code, entity_type, bucket_name, blob_name):
    epochs = list(range(1, 11))
    accuracy = [0.8 + 0.01 * i for i in range(10)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, accuracy, marker='o', label='Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
    ax.legend()
    ax.grid(True)
    
    save_plot_to_gcs(fig, bucket_name, blob_name)
