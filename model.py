import joblib
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

def train_and_predict(series, target_year):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=False)
    start_year = series.index[-1].year + 1
    end_year = target_year
    forecast = results.get_forecast(steps=target_year - start_year + 1)
    forecast_df = forecast.summary_frame()
    return forecast_df

def train_and_evaluate(series, eval_year=2021):
    # Supprimer les valeurs NaN par interpolation linéaire
    series = series.interpolate(method='linear').dropna()

    train_series = series[series.index.year < eval_year]
    test_series = series[series.index.year == eval_year]

    # Utiliser auto_arima pour trouver les meilleurs paramètres
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
