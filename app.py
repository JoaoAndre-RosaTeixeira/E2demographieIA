import os
import io
from flask import Flask, jsonify, request, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import joblib
from google.cloud import secretmanager, storage
from model import get_best_arima_model, plot_population_forecast, generate_monitoring_plot

db = SQLAlchemy()

def create_app(config=None):
    app = Flask(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    if config:
        app.config.update(config)

    with app.app_context():
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///:memory:')

        if 'sqlalchemy' not in app.extensions:
            db.init_app(app)
            db.create_all()

        flask_app = FlaskApp(app, db, project_id=app.config.get("GCP_PROJECT_ID", "dev-ia-e1"), 
                             bucket_name=app.config.get("GCS_BUCKET_NAME", "my-flask-app-bucket"))

        @app.route('/<entity_type>', methods=['GET'])
        def get_data(entity_type):
            return flask_app.get_data(entity_type)

        @app.route('/predict/<entity_type>', methods=['GET'])
        def predict(entity_type):
            return flask_app.predict(entity_type)

        @app.route('/get_plot/<entity_type>', methods=['GET'])
        def get_plot(entity_type):
            return flask_app.get_plot(entity_type)

        @app.route('/get_image', methods=['GET'])
        def get_image():
            return flask_app.get_image()

    return app

class Commune(db.Model):
    __tablename__ = 'communes'
    code = Column(String(10), primary_key=True)
    nom = Column(String(255), nullable=False)
    codes_postaux = Column(String)
    code_departement = Column(String(10), ForeignKey('departements.code'))
    code_region = Column(String(10), ForeignKey('regions.code'))
    populations = relationship('PopulationParAnnee', order_by='PopulationParAnnee.annee')

class Departement(db.Model):
    __tablename__ = 'departements'
    code = Column(String(10), primary_key=True)
    nom = Column(String(255), nullable=False)
    code_region = Column(String(10), ForeignKey('regions.code'))
    communes = relationship('Commune', backref='departement')

class Region(db.Model):
    __tablename__ = 'regions'
    code = Column(String(10), primary_key=True)
    nom = Column(String(255), nullable=False)
    departements = relationship('Departement', backref='region')

class PopulationParAnnee(db.Model):
    __tablename__ = 'population_par_annee'
    code_commune = Column(String(10), ForeignKey('communes.code'), primary_key=True)
    annee = Column(Integer, primary_key=True)
    population = Column(Integer)

entity_config = {
    'commune': {'model': Commune, 'code_attr': 'code', 'population_relationship': 'populations'},
    'departement': {'model': Departement, 'code_attr': 'code', 'population_relationship': 'communes'},
    'region': {'model': Region, 'code_attr': 'code', 'population_relationship': 'departements'}
}

class FlaskApp:
    def __init__(self, app, db, project_id, bucket_name):
        self.app = app
        self.db = db
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.secret_client = secretmanager.SecretManagerServiceClient()
        self.configure_db()

    def get_secret(self, secret_id):
        """Retrieve a secret from Google Cloud Secret Manager."""
        secret_name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
        response = self.secret_client.access_secret_version(name=secret_name)
        secret_value = response.payload.data.decode('UTF-8')
        return secret_value

    def configure_db(self):
        db_url = self.get_secret('database-url')
        self.app.config['SQLALCHEMY_DATABASE_URI'] = db_url
        if 'sqlalchemy' not in self.app.extensions:
            self.db.init_app(self.app)
            self.db.create_all()

    def save_to_gcs(self, model, destination_blob_name):
        """Serialize the model and upload to GCS."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        model_bytes = joblib.dumps(model)
        blob.upload_from_string(model_bytes)
        print(f"Model uploaded to {destination_blob_name}.")

    def load_from_gcs(self, source_blob_name):
        """Download the model from GCS and deserialize."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)
        model_bytes = blob.download_as_string()
        model = joblib.loads(model_bytes)
        print(f"Model downloaded from {source_blob_name}.")
        return model

    def save_plot_to_gcs(self, plot_bytes, destination_blob_name):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(plot_bytes.getvalue(), content_type='image/png')
        print(f"Plot uploaded to {destination_blob_name}.")

    def download_plot_from_gcs(self, source_blob_name):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)
        plot_bytes = blob.download_as_bytes()
        return io.BytesIO(plot_bytes)

    def get_data(self, entity_type):
        code = request.args.get('code', default=None)
        year = request.args.get('year', default=None)

        config = entity_config.get(entity_type)
        if not config:
            return jsonify(message="Invalid entity type"), 400

        model = config['model']
        query = self.db.session.query(model)

        if code:
            query = query.filter(getattr(model, config['code_attr']) == code)

        entities = query.all()
        if not entities:
            return jsonify(message="No entities found"), 404

        response = {}
        for entity in entities:
            if entity_type == 'commune':
                populations = entity.populations if year is None else [pop for pop in entity.populations if pop.annee == year]
            else:
                populations = []
                if entity_type == 'departement':
                    for commune in entity.communes:
                        populations.extend(commune.populations)
                elif entity_type == 'region':
                    for departement in entity.departements:
                        for commune in departement.communes:
                            populations.extend(commune.populations)

            population_summary = {}
            for pop in populations:
                if year and pop.annee != year:
                    continue
                if pop.annee in population_summary:
                    population_summary[pop.annee] += pop.population
                else:
                    population_summary[pop.annee] = pop.population

            response = {
                'code': entity.code,
                'nom': entity.nom,
                'populations': [{'annee': k, 'population': v} for k, v in sorted(population_summary.items())]
            }

            if entity_type == 'commune':
                response['codes_postaux'] = entity.codes_postaux

        return jsonify(response)

    def predict(self, entity_type):
        code = request.args.get('code')
        target_year = request.args.get('year')

        if not code or not target_year:
            return jsonify({'error': 'Veuillez fournir les paramètres "code" et "year".'}), 400

        try:
            target_year = int(target_year)
        except ValueError:
            return jsonify({'error': 'L\'année cible doit être un entier.'}), 400

        config = entity_config.get(entity_type)
        if not config:
            return jsonify(message="Invalid entity type"), 400

        model = config['model']
        entity = self.db.session.query(model).filter(getattr(model, config['code_attr']) == code).first()
        if not entity:
            return jsonify({'error': f'{entity_type.capitalize()} avec code {code} non trouvé.'}), 404

        populations = []
        if entity_type == 'commune':
            populations = entity.populations
        elif entity_type == 'departement':
            for commune in entity.communes:
                populations.extend(commune.populations)
        elif entity_type == 'region':
            for departement in entity.departements:
                for commune in departement.communes:
                    populations.extend(commune.populations)

        if not populations:
            return jsonify({'error': f'Aucune donnée de population trouvée pour ce {entity_type}.'}), 404

        # Préparation des données sous forme de série temporelle
        population_summary = {}
        for pop in populations:
            if pop.annee in population_summary:
                population_summary[pop.annee] += pop.population
            else:
                population_summary[pop.annee] = pop.population

        series = pd.Series(data=list(population_summary.values()), index=pd.to_datetime(list(population_summary.keys()), format='%Y')).asfreq('YS')
        series = series.interpolate(method='linear').dropna()  # Supprimer les NaN par interpolation linéaire

        # Vérifier si le modèle existe déjà
        model_filename = f"models/{entity_type}_{code}_{series.index[-1].year}.pkl"
        bucket = self.storage_client.bucket(self.bucket_name)
        if bucket.blob(model_filename).exists() and series.index[-1].year <= target_year:
            model = self.load_from_gcs(model_filename)
            print(f"Chargement du modèle existant pour {entity_type} avec code {code}.")
        else:
            # Entraîner et sauvegarder le modèle
            model = get_best_arima_model(series)
            self.save_to_gcs(model, model_filename)
            print(f"Entraînement et sauvegarde du nouveau modèle pour {entity_type} avec code {code}.")

        # Prédiction
        forecast_df = model.predict(n_periods=target_year - series.index[-1].year)
        forecast_index = pd.date_range(start=pd.to_datetime(f"{series.index[-1].year + 1}-01-01"), periods=len(forecast_df), freq='YS')
        forecast_df = pd.DataFrame(forecast_df, index=forecast_index, columns=['mean'])
        predicted_value = forecast_df['mean'].iloc[-1]

        # Sauvegarder le graphique
        plot_filename = f"plots/{entity_type}_{code}_{target_year}.png"
        plot_monitoring_filename = f"plots/monitoring_{entity_type}_{code}_{target_year}.png"
        
        plot_buffer = io.BytesIO()
        plot_population_forecast(series, forecast_df, plot_buffer)
        self.save_plot_to_gcs(plot_buffer, plot_filename)
        
        monitoring_buffer = io.BytesIO()
        generate_monitoring_plot(code, entity_type, monitoring_buffer)
        self.save_plot_to_gcs(monitoring_buffer, plot_monitoring_filename)

        # Construction de la réponse
        response = {
            'code': entity.code,
            'nom': entity.nom,
            'target_year': target_year,
            'predicted_population': int(predicted_value),
            'plot_url': f"/get_plot/{entity_type}?code={code}&year={target_year}",
        }

        if entity_type == 'commune':
            response['codes_postaux'] = entity.codes_postaux

        return jsonify(response)

    def get_plot(self, entity_type):
        code = request.args.get('code')
        year = request.args.get('year')

        if not code or not year:
            return jsonify({'error': 'Veuillez fournir les paramètres "code" et "year".'}), 400

        try:
            year = int(year)
        except ValueError:
            return jsonify({'error': 'L\'année doit être un entier.'}), 400

        config = entity_config.get(entity_type)
        if not config:
            return jsonify(message="Invalid entity type"), 400

        model = config['model']
        
        entity = self.db.session.query(model).filter(getattr(model, config['code_attr']) == code).first()
        if not entity:
            return jsonify({'error': f'{entity_type.capitalize()} avec code {code} non trouvé.'}), 404

        plot_filename = f"plots/{entity_type}_{code}_{year}.png"
        monitoring_filename = f"plots/monitoring_{entity_type}_{code}_{year}.png"

        bucket = self.storage_client.bucket(self.bucket_name)
        if not bucket.blob(plot_filename).exists():
            return jsonify({'error': 'Le graphique demandé n\'existe pas.'}), 404

        plot_buffer = self.download_plot_from_gcs(plot_filename)
        monitoring_buffer = self.download_plot_from_gcs(monitoring_filename)

        plot_url = f"/get_image?filename={plot_filename}"
        monitoring_url = f"/get_image?filename={monitoring_filename}"

        return jsonify({
            'plot_url': plot_url,
            'monitoring_url': monitoring_url
        })

    def get_image(self):
        filename = request.args.get('filename')

        if not filename:
            return jsonify({'error': 'Le fichier demandé n\'existe pas.'}), 404

        plot_buffer = self.download_plot_from_gcs(filename)
        return send_file(plot_buffer, mimetype='image/png')

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
