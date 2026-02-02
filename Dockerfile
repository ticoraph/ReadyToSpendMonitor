# Dockerfile pour l'API de scoring

FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier requirements et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY models/ ./models/  # Si vous avez des modèles pré-entraînés

# Création du répertoire de logs
RUN mkdir -p logs

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]