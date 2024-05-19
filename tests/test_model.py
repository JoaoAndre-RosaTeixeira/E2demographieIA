import pandas as pd
from model import get_best_arima_model, plot_population_forecast, generate_monitoring_plot
import io

def test_get_best_arima_model():
    series = pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS'))
    model = get_best_arima_model(series)
    assert model is not None

