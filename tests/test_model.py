import pandas as pd
from model import get_best_arima_model, plot_population_forecast, generate_monitoring_plot
import io
import matplotlib.pyplot as plt
from unittest.mock import patch

def test_get_best_arima_model():
    series = pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS'))
    model = get_best_arima_model(series)
    assert model is not None

@patch("model.save_plot_to_gcs")
def test_plot_population_forecast(mock_save_plot_to_gcs):
    series = pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS'))
    forecast_df = pd.DataFrame({'mean': [600, 700]}, index=pd.date_range("2025-01-01", periods=2, freq='YS'))
    plot_population_forecast(series, forecast_df, "test-bucket", "test-plot.png")
    assert mock_save_plot_to_gcs.called

@patch("model.save_plot_to_gcs")
def test_generate_monitoring_plot(mock_save_plot_to_gcs):
    cross_val_results = [
        {"metrics": {"rmse": 10, "mae": 8, "r2": 0.9}, "train_values": [100, 200], "test_values": [300], "predicted_values": [310]},
        {"metrics": {"rmse": 12, "mae": 10, "r2": 0.85}, "train_values": [200, 300], "test_values": [400], "predicted_values": [410]}
    ]
    generate_monitoring_plot("12345", "commune", cross_val_results, "test-bucket", "test-monitoring.png")
    assert mock_save_plot_to_gcs.called
