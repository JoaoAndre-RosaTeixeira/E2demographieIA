from urllib.parse import quote_plus
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from model import train_and_predict, train_and_evaluate, save_model, load_model, get_best_arima_model
import pandas as pd
import os

# Configuration de l'application Flask
app = Flask(__name__)
password = "^Te+7Qib&Q\"%@X>>"
encoded_password = quote_plus(password)
database_url = f'postgresql+psycopg2://postgres:{encoded_password}@34.155.209.64:5432/dev-ia-e1'
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialisation de SQLAlchemy avec Flask
db = SQLAlchemy(app)

# Définition des modèles
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

# Configuration des entités avec leurs attributs correspondants
entity_config = {
    'commune': {'model': Commune, 'code_attr': 'codes_postaux', 'population_relationship': 'populations'},
    'departement': {'model': Departement, 'code_attr': 'code', 'population_relationship': 'communes'},
    'region': {'model': Region, 'code_attr': 'code', 'population_relationship': 'departements'}
}

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

    results = []
    for entity in entities:
        entity_data = {
            'nom': entity.nom,
            'code': getattr(entity, config['code_attr']),
            'populations': []
        }

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

        entity_data['populations'] = [{'annee': k, 'population': v} for k, v in sorted(population_summary.items())]
        
        results.append(entity_data)

    return jsonify(results)

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
    model_filename = f"{entity_type}_{code}_{series.index[-1].year}.pkl"
    if os.path.exists(os.path.join('models', model_filename)) and series.index[-1].year <= target_year:
        model = load_model(f"models/{model_filename}")
        print(f"Chargement du modèle existant pour {entity_type} avec code {code}.")
    else:
        # Entraîner et sauvegarder le modèle
        model = get_best_arima_model(series)
        save_model(model, f"models/{model_filename}")
        print(f"Entraînement et sauvegarde du nouveau modèle pour {entity_type} avec code {code}.")

    # Prédiction
    forecast_df = model.predict(n_periods=target_year - series.index[-1].year)
    forecast_index = pd.date_range(start=pd.to_datetime(f"{series.index[-1].year + 1}-01-01"), periods=len(forecast_df), freq='YS')
    forecast_df = pd.DataFrame(forecast_df, index=forecast_index, columns=['mean'])
    predicted_value = forecast_df['mean'].iloc[-1]

    # Construction de la réponse
    response = {
        'code': code,
        'nom': entity.nom,
        'target_year': target_year,
        'predicted_population': int(predicted_value)
    }

    if entity_type == 'commune':
        response['codes_postaux'] = entity.codes_postaux

    return jsonify(response)

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

    # Préparation des données sous forme de série temporelle
    population_summary = {}
    for pop in populations:
        if pop.annee in population_summary:
            population_summary[pop.annee] += pop.population
        else:
            population_summary[pop.annee] = pop.population


    series = pd.Series(data=list(population_summary.values()), index=pd.to_datetime(list(population_summary.keys()), format='%Y')).asfreq('YS')
    series = series.interpolate(method='linear').dropna()  # Supprimer les NaN par interpolation linéaire

    # Déterminer la dernière année dans les données pour l'évaluation
    eval_year = series.index[-1].year

    # Vérifier si le modèle existe déjà
    model_filename = f"{entity_type}_{code}_{eval_year}.pkl"
    if os.path.exists(os.path.join('train_models', model_filename)) and series.index[-1].year <= eval_year:
        model = load_model(f"train_models/{model_filename}")
        print(f"Chargement du modèle existant pour {entity_type} avec code {code}.")
        return jsonify({'message': f'Modèle chargé depuis {model_filename}.'})
    else:
        # Entraînement et évaluation
        accuracy, best_order, best_seasonal_order = train_and_evaluate(series, eval_year=eval_year)

        # Sauvegarder le modèle
        model = get_best_arima_model(series)
        save_model(model, f"train_models/{model_filename}")
        print(f"Modèle sauvegardé sous {model_filename}.")

    # Construction de la réponse
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

if __name__ == '__main__':
    app.run(debug=True)
