import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# Charger les données
data_path = 'communes.csv'
communes_data = pd.read_csv(data_path, dtype={'code': str, 'code_region': str, 'code_departement': str}, low_memory=False)

# Nettoyer les données
for column in communes_data.columns:
    if "Population en" in column:
        median_value = communes_data[communes_data[column] != 0][column].median()
        communes_data[column] = communes_data[column].replace(0, median_value)
        communes_data[column] = pd.to_numeric(communes_data[column], errors='coerce')

# Transformation des données pour la modélisation
communes_long = communes_data.melt(id_vars=['code', 'code_region', 'code_departement', 'nom', 'codes_postaux'],
                                   value_vars=[col for col in communes_data if col.startswith('Population en')],
                                   var_name='Annee',
                                   value_name='Population')
communes_long['Annee'] = communes_long['Annee'].apply(lambda x: int(x.split(' ')[-1]))

# Encodage des variables catégorielles
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(communes_long[['code_region', 'code_departement']]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['code_region', 'code_departement']))
encoded_df.index = communes_long.index
communes_long = pd.concat([communes_long, encoded_df], axis=1)

# Divisez les données en ensembles d'entraînement et de test
X = communes_long.drop('Population', axis=1)
y = communes_long['Population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train.loc[:, X_train.columns.str.startswith('feature_') | (X_train.columns == 'Annee')] = scaler.fit_transform(X_train.loc[:, X_train.columns.str.startswith('feature_') | (X_train.columns == 'Annee')])

# Sélection de caractéristiques
selector = SelectKBest(score_func=f_regression, k=5)
X_train.loc[:, X_train.columns.str.startswith('feature_') | (X_train.columns == 'Annee')] = selector.fit_transform(X_train.loc[:, X_train.columns.str.startswith('feature_') | (X_train.columns == 'Annee')], y_train)

# Utilisation d'un modèle plus complexe
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Prédiction pour Vaugneray
data_commune = {'code': '69255', 'code_region': '84', 'code_departement': '69', 'nom': 'Vaugneray', 'codes_postaux': '69670', 'Annee': 2022}
X_pred = pd.DataFrame(data_commune, index=[0])
X_pred_encoded = encoder.transform(X_pred[['code_region', 'code_departement']])
X_pred_encoded_df = pd.DataFrame(X_pred_encoded, columns=[f"feature_{i}" for i in range(X_pred_encoded.shape[1])])
X_pred = pd.concat([X_pred, X_pred_encoded_df], axis=1)

X_pred_scaled = scaler.transform(X_pred.loc[:, X_pred.columns.str.startswith('feature_') | (X_pred.columns == 'Annee')])
X_pred_selected = selector.transform(X_pred_scaled)

y_pred = rf_reg.predict(X_pred_selected)

print(f"Population prédite pour la commune {data_commune['nom']} en {data_commune['Annee']}: {y_pred[0]:.2f}")

# Calcul et affichage des métriques de précision
y_pred_train = rf_reg.predict(X_train)

mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
rmse = np.sqrt(mse)
r_squared = r2_score(y_train, y_pred_train)
mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
print(f"MAPE: {mape:.2f}%")