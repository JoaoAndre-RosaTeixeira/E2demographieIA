import os
from flask import Flask, jsonify, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import joblib
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from google.cloud import secretmanager, storage
from model import download_from_gcs, plot_population_forecast, generate_monitoring_plot, get_best_arima_model, save_to_gcs, train_and_evaluate, save_model_info_to_gcs, load_model_info_from_gcs

def get_secret(secret_id, project_id):
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=secret_name)
    secret_value = response.payload.data.decode('UTF-8')
    return secret_value

bucket_name = 'my-flask-app-bucket'
project_id = "dev-ia-e1"
db_url = get_secret('database-url', project_id)

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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
    'commune': {'model': Commune, 'code_attr': 'code', 'population_relationship': 'populations', 'entity_code_relationship': Commune.code_departement},
    'departement': {'model': Departement, 'code_attr': 'code', 'population_relationship': 'communes', 'entity_code_relationship': Departement.code_region},
    'region': {'model': Region, 'code_attr': 'code', 'population_relationship': 'departements', 'entity_code_relationship': Region.code}
}

@app.route('/train/<entity_type>', methods=['GET'])
def train(entity_type):
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Veuillez fournir le paramètre "code".'}), 400
    config = entity_config.get(entity_type)
    if not config:
        return jsonify(message="Invalid entity type"), 400
    model = config['model']
    entity = db.session.query(model).filter(getattr(model, config['code_attr']) == code).first()
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
    population_summary = {}
    for pop in populations:
        if pop.annee in population_summary:
            population_summary[pop.annee] += pop.population
        else:
            population_summary[pop.annee] = pop.population
    series = pd.Series(data=list(population_summary.values()), index=pd.to_datetime(list(population_summary.keys()), format='%Y')).asfreq('YS')
    series = series.interpolate(method='linear').dropna()
    eval_year = series.index[-1].year
    model_filename = f"{entity_type}_{code}_{eval_year}.pkl"
    model_info_filename = f"train_models/{entity_type}_{code}_{eval_year}_info.json"
    if os.path.exists(os.path.join('train_models', model_filename)) and os.path.exists(model_info_filename) and series.index[-1].year <= eval_year:
        model = joblib.load(f"train_models/{model_filename}")
        model_info = load_model_info_from_gcs(bucket_name, model_info_filename)
        accuracy = model_info['accuracy']
        best_order = model_info['best_order']
        best_seasonal_order = model_info['best_seasonal_order']
        print(f"Chargement du modèle existant pour {entity_type} avec code {code}.")
    else:
        accuracy, best_order, best_seasonal_order = train_and_evaluate(series, eval_year=eval_year)
        model = get_best_arima_model(series)
        joblib.dump(model, f"train_models/{model_filename}")
        save_to_gcs(bucket_name, f"train_models/{model_filename}", model_filename)
        save_model_info_to_gcs({
            'accuracy': accuracy,
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order
        }, bucket_name, model_info_filename)
        print(f"Modèle sauvegardé sous {model_filename} avec les informations d'évaluation {model_info_filename}.")
    response = {
        'code': code,
        'nom': entity.nom,
        'accuracy': accuracy,
        'best_order': best_order,
        'best_seasonal_order': best_seasonal_order
    }
    if entity_type == 'commune':
        response['codes_postaux'] = entity.codes_postaux
    return jsonify(response)

@app.route('/<entity_type>', methods=['GET'])
def get_data(entity_type):
    code = request.args.get('code', default=None)
    year = request.args.get('year', default=None)
    config = entity_config.get(entity_type)
    if not config:
        return jsonify(message="Invalid entity type"), 400
    model = config['model']
    query = db.session.query(model)
    if code:
        query = query.filter(getattr(model, config['code_attr']) == code)
    entities = query.all()
    if not entities:
        return jsonify(message="No entities found"), 404
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

@app.route('/form/<entity_type>', methods=['GET'])
def get_entity(entity_type):
    config = entity_config.get(entity_type)
    code = request.args.get('code')
    if not config:
        return jsonify(message="Invalid entity type"), 400
    model = config['model']
    entitys = db.session.query(model).filter(config['entity_code_relationship'] == code).all() if code and model != Region else db.session.query(model).all()
    response = [{'code': entity.code, 'nom': entity.nom} for entity in entitys]
    return jsonify(response)

@app.route('/predict/<entity_type>', methods=['GET'])
def predict(entity_type):
    code = request.args.get('code')
    target_year = request.args.get('year')
    if not code or not target_year:
        return jsonify({'error': 'Veuillez fournir les paramètres "code" et "target_year".'}), 400
    try:
        target_year = int(target_year)
    except ValueError:
        return jsonify({'error': 'L\'année cible doit être un entier.'}), 400
    config = entity_config.get(entity_type)
    if not config:
        return jsonify(message="Invalid entity type"), 400
    model = config['model']
    entity = db.session.query(model).filter(getattr(model, config['code_attr']) == code).first()
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
    population_summary = {}
    for pop in populations:
        if pop.annee in population_summary:
            population_summary[pop.annee] += pop.population
        else:
            population_summary[pop.annee] = pop.population
    series = pd.Series(data=list(population_summary.values()), index=pd.to_datetime(list(population_summary.keys()), format='%Y')).asfreq('YS')
    series = series.interpolate(method='linear').dropna()
    model_filename = f"{entity_type}_{code}_{series.index[-1].year}.pkl"
    blob_path = f"models/{model_filename}"
    model_info_filename = f"train_models/{entity_type}_{code}_{series.index[-1].year}_info.json"
    bucket = storage.Client().bucket(bucket_name)
    if bucket.blob(blob_path).exists():
        download_from_gcs(bucket_name, blob_path, model_filename)
        model = joblib.load(model_filename)
        model_info = load_model_info_from_gcs(bucket_name, model_info_filename)
        accuracy = model_info['accuracy']
        best_order = model_info['best_order']
        best_seasonal_order = model_info['best_seasonal_order']
        print(f"Chargement du modèle existant pour {entity_type} avec code {code}.")
    else:
        accuracy, best_order, best_seasonal_order = train_and_evaluate(series, eval_year=series.index[-1].year)
        model = get_best_arima_model(series)
        joblib.dump(model, model_filename)
        save_to_gcs(bucket_name, model_filename, blob_path)
        save_model_info_to_gcs({
            'accuracy': accuracy,
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order
        }, bucket_name, model_info_filename)
        print(f"Entraînement et sauvegarde du nouveau modèle pour {entity_type} avec code {code}.")
    forecast_df = model.predict(n_periods=target_year - series.index[-1].year)
    forecast_index = pd.date_range(start=pd.to_datetime(f"{series.index[-1].year + 1}-01-01"), periods=len(forecast_df), freq='YS')
    forecast_df = pd.DataFrame(forecast_df, index=forecast_index, columns=['mean'])
    predicted_value = forecast_df['mean'].iloc[-1]
    plot_filename = f"plots/{entity_type}_{code}_{target_year}.png"
    plot_population_forecast(series, forecast_df, bucket_name, plot_filename)
    plot_monitoring_filename = f"plots/monitoring_{entity_type}_{code}_{target_year}.png"
    generate_monitoring_plot(code, entity_type, [accuracy], bucket_name, plot_monitoring_filename)
    plot_url = f"https://storage.googleapis.com/{bucket_name}/{plot_filename}"
    monitoring_url = f"https://storage.googleapis.com/{bucket_name}/{plot_monitoring_filename}"
    response = {
        'code': entity.code,
        'nom': entity.nom,
        'target_year': target_year,
        'accuracy': accuracy,
        'predicted_population': int(predicted_value),
        'plot_url': plot_url,
        'monitoring_url': monitoring_url
    }
    if entity_type == 'commune':
        response['codes_postaux'] = entity.codes_postaux
    return jsonify(response)

@app.route('/get_image', methods=['GET'])
def get_image():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Le fichier demandé n\'existe pas.'}), 404
    return redirect(filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)