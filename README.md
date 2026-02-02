# ReadyToSpendMonitor

Mise en production d'un modèle de scoring pour l'entreprise "Prêt à Dépenser".

Ce projet inclut la création d'une API robuste, la conteneurisation pour un déploiement fluide, et la mise en place d'un monitoring proactif pour garantir la performance et la fiabilité du modèle dans le temps.

## 📋 Contenu

- API FastAPI pour les prédictions de scoring
- Conteneurisation Docker
- Pipeline CI/CD (GitHub Actions)
- Tests automatisés
- Monitoring et détection du data drift
- Dashboard Streamlit

## 🏗️ Structure du projet

```

```

## 🚀 Installation

### Prérequis

- Python 3.10
- Docker
- Git

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/ticoraph/ReadyToSpendMonitor
cd ReadyToSpendMonitor

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement (optionnel)
cp .env.example .env
```

### Installation du modèle

Placez votre fichier de modèle entraîné dans le répertoire `models/` avec le nom `scoring_model.pkl`.

## 🏃 Lancer l'API

### En local

```bash
# Lancer l'API avec uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera accessible sur http://localhost:8000

- Documentation Swagger: http://localhost:8000/docs
- Endpoint de santé: http://localhost:8000/health

### Avec Docker

```bash
# Construire l'image Docker
docker build -t scoring-api .

# Lancer le conteneur
docker run -p 8000:8000 -v $(pwd)/models:/app/models scoring-api
```

### Avec Docker Compose

```bash
cd docker
docker-compose up -d
```

## 🧪 Exécuter les tests

```bash
# Exécuter tous les tests
pytest

# Exécuter avec couverture
pytest --cov=src --cov-report=html

# Exécuter un test spécifique
pytest tests/test_api.py -v
```

## 📊 Monitoring

### Dashboard Streamlit

```bash
streamlit run notebooks/dashboard.py
```

Le dashboard affiche :
- Distribution des scores prédits
- Latence de l'API
- Temps d'inférence
- Analyse du data drift

### Logs

Les logs sont stockés dans `logs/` :
- `api.log`: Logs de l'API
- `predictions.csv`: Données des prédictions pour l'analyse du drift

## 🔄 Pipeline CI/CD

Le pipeline GitHub Actions automatise :
1. Exécution des tests à chaque push
2. Construction de l'image Docker
3. Déploiement sur la branche main

## 📝 Interprétation du monitoring

### Distribution des scores
- **Score 0**: Client à faible risque
- **Score 1**: Client à haut risque

### Métriques clés
| Métrique | Description | Seuil d'alerte |
|----------|-------------|----------------|
| Latence | Temps de réponse de l'API | > 500ms |
| Temps d'inférence | Temps de calcul du modèle | > 100ms |
| Taux d'erreur | Requêtes en échec | > 5% |
| Drift | Écart distribution des données | > 0.3 |

## 🔧 Configuration

La configuration se fait via variables d'environnement ou fichier `.env` :

```bash
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=models
```

## 📚 Documentation

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Docker](https://docs.docker.com/)
- [Documentation Evidently](https://docs.evidentlyai.com/)