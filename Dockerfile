# Utiliser une image de base officielle de Python 3.12
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel Flask va tourner
EXPOSE 8080

# Définir la variable d'environnement pour dire à Flask d'écouter sur toutes les IPs
ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT 8080

# Commande pour lancer l'application
ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=8080"]
