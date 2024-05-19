import pandas as pd
from model import get_best_arima_model, plot_population_forecast, generate_monitoring_plot

def test_get_best_arima_model():
    series = pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS'))
    model = get_best_arima_model(series)
    assert model is not None


def test_plot_population_forecast():
    series = pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS'))
    forecast_df = pd.DataFrame({'mean': [150, 250, 350]}, index=pd.date_range("2025-01-01", periods=3, freq='YS'))
    buffer = io.BytesIO()
    plot_population_forecast(series, forecast_df, buffer)
    assert buffer.tell() > 0  # Check if the buffer has content

def test_generate_monitoring_plot():
    buffer = io.BytesIO()
    generate_monitoring_plot("123", "commune", buffer)
    assert buffer.tell() > 0  # Check if the buffer has content
