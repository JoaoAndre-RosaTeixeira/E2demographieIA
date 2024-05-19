import sys
import os
import pytest
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app, db as _db


@pytest.fixture(scope='module')
def app():
    with patch('app.storage.Client') as MockStorageClient, \
         patch('app.secretmanager.SecretManagerServiceClient') as MockSecretManagerClient:
        
        MockStorageClient.return_value = MockStorageClient
        MockSecretManagerClient.return_value = MockSecretManagerClient

        app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'GCP_PROJECT_ID': 'test-project',
            'GCS_BUCKET_NAME': 'test-bucket'
        })

        with app.app_context():
            _db.create_all()
            yield app
            _db.drop_all()

@pytest.fixture(scope='module')
def client(app):
    return app.test_client()

@pytest.fixture(scope='function')
def db(app):
    with app.app_context():
        _db.drop_all()
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()
