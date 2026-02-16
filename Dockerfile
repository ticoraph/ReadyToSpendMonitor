# Utiliser une image Python légère
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y libgomp1 curl supervisor && \
    rm -rf /var/lib/apt/lists/*

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
COPY scripts scripts
COPY tests tests

# Créer les dossiers nécessaires
RUN mkdir -p logs

# Configurer Supervisor pour gérer les processus
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exposer les ports
EXPOSE 8000 8501

# Lancer Supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]