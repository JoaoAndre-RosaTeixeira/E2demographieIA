import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense
import io

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres pour ARIMA
def get_best_arima_model(series):
    param_grid = {
        'order': [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2)],
        'seasonal_order': [(0, 0, 0, 0), (1, 0, 0, 12), (0, 1, 0, 12), (1, 1, 0, 12)]
    }
    model = auto_arima(seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(series)
    best_params = grid_search.best_params_
    print(f"Meilleure configuration trouvée : {best_params}")
    best_model = grid_search.best_estimator_
    return best_model

# Fonction pour traiter les valeurs aberrantes
def remove_outliers(series):
    z_scores = np.abs(stats.zscore(series))
    return series[z_scores < 3]


# Fonction pour créer et entraîner un modèle de réseau de neurones
def train_neural_network(series):
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(series.values, series.index, epochs=50, batch_size=10)
    return model

# Fonction de validation croisée
def perform_cross_validation(series, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []
    for train_index, test_index in tscv.split(series):
        train, test = series.iloc[train_index], series.iloc[test_index]
        model = get_best_arima_model(train)
        y_pred = model.predict(n_periods=len(test))
        accuracy = calculate_rmse(test.values, y_pred)
        accuracies.append(accuracy)
    return accuracies

# Fonction pour calculer RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Fonction pour générer le plot de monitoring
def generate_monitoring_plot(code, entity_type, accuracies, bucket_name, blob_name):
    splits = list(range(1, len(accuracies) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(splits, accuracies, marker='o', label='RMSE')
    ax.set_xlabel('Splits')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
    ax.legend()
    ax.grid(True)
    save_plot_to_gcs(fig, bucket_name, blob_name)
    
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


# Fonction pour sauvegarder les plots sur GCS
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

def save_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}.")


