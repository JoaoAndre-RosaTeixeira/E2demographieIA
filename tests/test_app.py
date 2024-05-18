import pytest
import sys
import os
import pandas as pd
from pmdarima import auto_arima

# Ajouter le répertoire racine au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import get_best_arima_model, save_model, load_model, train_and_evaluate

@pytest.fixture
def sample_series():
    dates = pd.date_range(start='2000-01-01', periods=20, freq='YS')
    data = [x * 100 for x in range(20)]
    return pd.Series(data=data, index=dates)

def test_get_best_arima_model(sample_series):
    model = get_best_arima_model(sample_series)
    assert model is not None

def test_save_and_load_model(sample_series):
    model = get_best_arima_model(sample_series)
    save_model(model, 'test_model.pkl')
    loaded_model = load_model('test_model.pkl')
    assert model.order == loaded_model.order

def test_train_and_evaluate(sample_series):
    accuracy, best_order, best_seasonal_order = train_and_evaluate(sample_series, 2018)
    assert accuracy is not None
    assert isinstance(best_order, tuple)
    assert isinstance(best_seasonal_order, tuple)
