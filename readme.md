#### Introduction
Ce projet utilise des modèles ARIMA pour prédire la population des communes françaises et le nombre de logements nécessaires. L'application est déployée sur Google Cloud Platform (GCP) et expose une API via Flask.

#### Installation et Configuration

1. **Cloner le dépôt** :
   ```bash
   git clone <url-du-depot>
   cd <nom-du-dossier>
   ```

2. **Configurer l'environnement virtuel** :
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Pour Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement** :
   Créez un fichier `.env` et ajoutez les variables suivantes :
   ```env
   DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<database>
   GCP_PROJECT_ID=<votre-project-id>
   GCS_BUCKET_NAME=<votre-bucket-name>
   ```

5. **Configurer Google Cloud** :
   - Configurez l'authentification avec Google Cloud SDK sur votre machine.
   - Créez un bucket Google Cloud Storage et configurez les permissions nécessaires.

#### Exécution de l'Application

1. **Initialiser la base de données** :
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

2. **Lancer l'application localement** :
   ```bash
   flask run
   ```

3. **Tester l'API** :
   Utilisez un outil comme Postman pour envoyer des requêtes à l'API. Par exemple :
   ```
   GET http://127.0.0.1:5000/predict/commune?code=01004&year=2025
   ```

#### Déploiement sur Google Cloud Platform

1. **Construire l'image Docker** :
   ```bash
   docker build -t gcr.io/<votre-project-id>/<nom-de-l-image> .
   ```

2. **Pousser l'image sur Google Container Registry (GCR)** :
   ```bash
   docker push gcr.io/<votre-project-id>/<nom-de-l-image>
   ```

3. **Déployer sur Google Cloud Run** :
   ```bash
   gcloud run deploy <nom-du-service> --image gcr.io/<votre-project-id>/<nom-de-l-image> --platform managed
   ```

#### Monitorage

1. **Importance du Monitorage** :
   - Maintenir la précision des prédictions.
   - Détecter les anomalies dans les données.
   - Optimiser les modèles en fonction des retours.

2. **Outils de Monitorage Utilisés** :
   - Génération de graphiques pour visualiser l'accuracy et d'autres métriques.
   - Sauvegarde des graphiques sur Google Cloud Storage pour une consultation facile.

3. **Exemple de Monitorage** :
   - À chaque prédiction, un graphique de monitorage est généré et sauvegardé.

#### Tests

1. **Configurer les Tests** :
   - Assurez-vous que les tests sont configurés dans le dossier `tests`.

2. **Exécuter les Tests** :
   ```bash
   pytest tests
   ```

3. **Types de Tests** :
   - **Tests Unitaires** : Valident les composants individuels de l'application.
   - **Tests d'Intégration** : Vérifient l'interaction entre les différents composants.

#### Chaîne de Livraison Continue (CI/CD)

1. **Configuration de GitHub Actions** :
   - Un pipeline CI/CD utilisant GitHub Actions pour automatiser le processus de build et de déploiement.

2. **Déclenchement Automatique** :
   - Le pipeline est configuré pour se déclencher automatiquement à chaque push ou pull request vers la branche principale.
