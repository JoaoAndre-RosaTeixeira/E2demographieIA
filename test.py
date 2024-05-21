import psycopg2
import urllib.parse as up
from google.cloud import secretmanager, storage


def get_secret(secret_id, project_id):
    """Retrieve a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=secret_name)
    secret_value = response.payload.data.decode('UTF-8')
    return secret_value

def test_db_connection(db_url):
    try:
        # Reformater l'URL pour psycopg2
        up.uses_netloc.append("postgres")
        url = up.urlparse(db_url)
        
        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        print("Connection successful")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")

# Exemple d'utilisation
project_id = "dev-ia-e1"
db_url = get_secret('database-url', project_id)
print(f"Database URL: {db_url}")  # Imprimez l'URL pour vérifier son format
test_db_connection(db_url)
