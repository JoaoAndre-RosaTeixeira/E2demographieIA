import pytest
from app import create_app, db
from unittest.mock import patch

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
            db.create_all()
            yield app
            db.drop_all()

@pytest.fixture(scope='module')
def client(app):
    return app.test_client()
