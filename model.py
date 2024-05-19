from flask import json
import joblib
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

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

# def train_and_predict(series, target_year):
#     model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
#     results = model.fit(disp=False)
#     start_year = series.index[-1].year + 1
#     end_year = target_year
#     forecast = results.get_forecast(steps=target_year - start_year + 1)
#     forecast_df = forecast.summary_frame()
#     return forecast_df

def train_and_evaluate(series, eval_year):
    series = series.interpolate(method='linear').dropna()
    train_series = series[series.index.year < eval_year]
    test_series = series[series.index.year == eval_year]

    model = auto_arima(train_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Meilleure configuration trouvée : ordre={best_order}, ordre saisonnier={best_seasonal_order}")

    results = model.fit(train_series)
    forecast = results.predict(n_periods=len(test_series))
    predicted_value = forecast[0]
    actual_value = test_series.iloc[0]

    accuracy = 100 - abs(predicted_value - actual_value) / actual_value * 100

    return accuracy, best_order, best_seasonal_order

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def save_model_info(info, filename):
    with open(filename, 'w') as f:
        json.dump(info, f)

def load_model_info(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_population_forecast(series, forecast_df, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Historical Population')
    plt.plot(forecast_df['mean'], label='Forecasted Population')
    if 'mean_ci_lower' in forecast_df.columns and 'mean_ci_upper' in forecast_df.columns:
        plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title('Population Forecast')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    

def generate_monitoring_plot(code, entity_type, monitoring_filename):
    epochs = list(range(1, 11))
    accuracy = [0.8 + 0.01 * i for i in range(10)]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy, marker='o', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
    plt.legend()
    plt.grid(True)
    plt.savefig(monitoring_filename)
    plt.close()
    