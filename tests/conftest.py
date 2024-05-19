import pytest
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app import app as flask_app, db as flask_db, FlaskApp

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def db(app):
    with app.app_context():
        flask_db.create_all()
        yield flask_db
        flask_db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()
