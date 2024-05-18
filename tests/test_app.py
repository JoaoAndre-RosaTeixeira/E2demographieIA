import pytest
from api import app, db

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # Utiliser une base de données en mémoire pour les tests
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to Flask" in response.data

def test_get_data(client):
    response = client.get('/commune?code=01004')
    assert response.status_code == 200
    assert 'code' in response.json
    assert 'nom' in response.json
