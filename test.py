import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Charger les données
data = pd.read_csv('path_to_your_dataset.csv')

# Séparer les caractéristiques et la cible
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèles
models = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"{name} Performance:")
    print(f"MAE: {mean_absolute_error(y_test, predictions)}")
    print(f"MAPE: {np.mean(np.abs((y_test - predictions) / y_test)) * 100}")
    print(f"MSE: {mean_squared_error(y_test, predictions)}")
    print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False)}")
    print(f"R-squared: {r2_score(y_test, predictions)}\n")
