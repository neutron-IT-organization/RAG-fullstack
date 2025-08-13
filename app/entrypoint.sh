#!/bin/bash
set -e

echo "--- Installation des dépendances ---"
pip install --no-cache-dir -r /app/requirements.txt

echo "--- Lancement du téléchargement des modèles ---"
python -u /app/download_models.py

echo "--- Démarrage de l'application Flask ---"
exec gunicorn --workers 2 --bind 0.0.0.0:8080 app_flask:app
