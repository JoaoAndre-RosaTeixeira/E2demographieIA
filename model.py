from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd
import io

def get_best_arima_model(series):
    model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    best_seasonal_order = model.seasonal_order
    print(f"Meilleure configuration trouvée : ordre={best_order}, ordre saisonnier={best_seasonal_order}")
    return model

def plot_population_forecast(series, forecast_df, buffer):
    try:
        print("Generating plot_population_forecast")
        print(f"Series: {series}")
        print(f"Forecast: {forecast_df}")

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

        # Save the plot to the buffer and flush
        plt.savefig(buffer, format='png')
        buffer.flush()
        buffer.seek(0)  # Rewind the buffer to the beginning
        plt.close()

        print(f"Generated plot_population_forecast with buffer size: {buffer.tell()}")
    except Exception as e:
        print(f"Error generating plot_population_forecast: {e}")

def generate_monitoring_plot(code, entity_type, buffer):
    try:
        print("Generating generate_monitoring_plot")
        epochs = list(range(1, 11))
        accuracy = [0.8 + 0.01 * i for i in range(10)]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, accuracy, marker='o', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Monitoring des Performances du Modèle pour {entity_type} {code}')
        plt.legend()
        plt.grid(True)

        # Save the plot to the buffer and flush
        plt.savefig(buffer, format='png')
        buffer.flush()
        buffer.seek(0)  # Rewind the buffer to the beginning
        plt.close()

        print(f"Generated generate_monitoring_plot with buffer size: {buffer.tell()}")
    except Exception as e:
        print(f"Error generating generate_monitoring_plot: {e}")
