# Utiliser une image Python légère
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code 
COPY api api
COPY models models
COPY monitoring monitoring
COPY tests tests

# Créer les dossiers nécessaires
RUN mkdir -p logs output

# Exposer le port 8000
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]