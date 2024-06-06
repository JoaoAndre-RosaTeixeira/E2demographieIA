import io
import json
import pandas as pd
from unittest.mock import patch
from model import get_best_arima_model

def test_get_data(client, db):
    from app import Commune, PopulationParAnnee
    commune = Commune(code="12345", nom="Test Commune")
    db.session.add(commune)
    db.session.commit()
    population = PopulationParAnnee(code_commune="12345", annee=2020, population=1000)
    db.session.add(population)
    db.session.commit()

    response = client.get('/commune?code=12345')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert data[0]['code'] == "12345"
    assert data[0]['nom'] == "Test Commune"

@patch("model.download_from_gcs")
@patch("model.save_to_gcs")
def test_predict(mock_save_to_gcs, mock_download_from_gcs, client, db):
    from app import Commune, PopulationParAnnee
    commune = Commune(code="12345", nom="Test Commune")
    db.session.add(commune)
    db.session.commit()
    for year in range(2010, 2021):
        population = PopulationParAnnee(code_commune="12345", annee=year, population=1000 + 10 * (year - 2010))
        db.session.add(population)
    db.session.commit()

    mock_model = get_best_arima_model(pd.Series([100, 200, 300, 400, 500], index=pd.date_range("2020-01-01", periods=5, freq='YS')))
    mock_download_from_gcs.return_value = mock_model

    response = client.get('/predict/commune?code=12345&year=2030')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'predicted_population' in data

@patch("model.download_from_gcs")
def test_get_image(mock_download_from_gcs, client, db):
    from app import Commune, PopulationParAnnee
    commune = Commune(code="12345", nom="Test Commune")
    db.session.add(commune)
    db.session.commit()
    for year in range(2010, 2021):
        population = PopulationParAnnee(code_commune="12345", annee=year, population=1000 + 10 * (year - 2010))
        db.session.add(population)
    db.session.commit()

    mock_download_from_gcs.return_value = io.BytesIO(b"fake image data")

    response = client.get('/get_image?filename=test-plot.png')
    assert response.status_code == 302  # Redirect status
